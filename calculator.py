import pulp
import json
from collections import OrderedDict, defaultdict
from graphlib import TopologicalSorter

class FactoryOptimizer:
    def __init__(self, recipes, rate_multiplier, factory_level, total_factories):
        self.recipes = recipes
        self.rate_multiplier = rate_multiplier
        self.factory_level = factory_level
        self.total_factories = total_factories
        self._clean_recipes()
        self._precalculate_rates()
        self._build_dependency_graph()

    def _clean_recipes(self):
        if "Barrel to Leek" in self.recipes:
            del self.recipes["Barrel to Leek"]

    def _precalculate_rates(self):
        for name, data in self.recipes.items():
            base_rate = data["Base Rate"]
            effective_rate = base_rate * (self.rate_multiplier ** (self.factory_level - 1))
            data["Effective Rate"] = effective_rate
            data["GPM"] = effective_rate * data["Gold Gain"] * 60

    def _build_dependency_graph(self):
        self.dependency_graph = {}
        for name, data in self.recipes.items():
            dependencies = []
            for input_res in data["Inputs"]:
                if input_res in self.recipes:
                    dependencies.append(input_res)
            self.dependency_graph[name] = set(dependencies)
        self.build_order = list(TopologicalSorter(self.dependency_graph).static_order())

    def optimize_gpm(self):
        prob = pulp.LpProblem("GPM_Optimization", pulp.LpMaximize)
        vars = pulp.LpVariable.dicts("F", self.recipes.keys(), lowBound=0, cat="Integer")
        
        resource_balance = defaultdict(pulp.LpAffineExpression)
        for name in self.build_order:
            data = self.recipes[name]
            rate = data["Effective Rate"]
            count = vars[name]
            
            for res, qty in data["Outputs"].items():
                resource_balance[res] += qty * rate * count
            for res, qty in data["Inputs"].items():
                if res != "Leek":
                    resource_balance[res] -= qty * rate * count

        prob += pulp.lpSum(vars[name] * data["GPM"] for name, data in self.recipes.items())
        prob += pulp.lpSum(vars.values()) <= self.total_factories
        
        for res in resource_balance:
            if res != "Leek":
                prob += resource_balance[res] >= 0

        if prob.solve(pulp.PULP_CBC_CMD(msg=0)) != pulp.LpStatusOptimal:
            return None
            
        return self._parse_results(vars, "GPM")

    def optimize_tiers(self):
        for target_tier in reversed(self.build_order):
            prob = pulp.LpProblem("Tier_Optimization", pulp.LpMaximize)
            vars = pulp.LpVariable.dicts("F", self.recipes.keys(), lowBound=0, cat="Integer")
            
            prob += vars[target_tier] * self.recipes[target_tier]["Effective Rate"]
            prob += pulp.lpSum(vars.values()) <= self.total_factories
            
            resource_balance = defaultdict(pulp.LpAffineExpression)
            for name in self.build_order:
                data = self.recipes[name]
                rate = data["Effective Rate"]
                count = vars[name]
                
                for res, qty in data["Outputs"].items():
                    resource_balance[res] += qty * rate * count
                for res, qty in data["Inputs"].items():
                    if res != "Leek":
                        resource_balance[res] -= qty * rate * count

            for res in resource_balance:
                if res != "Leek":
                    prob += resource_balance[res] >= 0

            if prob.solve(pulp.PULP_CBC_CMD(msg=0)) == pulp.LpStatusOptimal:
                if vars[target_tier].varValue > 0:
                    return self._parse_results(vars, "Tier", target_tier)
        return None

    def _parse_results(self, vars, mode, target_tier=None):
        results = {
            "allocations": [],
            "total_gpm": 0,
            "mode": mode,
            "target_tier": target_tier
        }
        
        allocs = []
        resource_flows = defaultdict(float)
        for name in self.build_order:
            data = self.recipes[name]
            count = int(vars[name].varValue + 0.5)
            if count <= 0:
                continue
                
            rate = data["Effective Rate"]
            production = rate * count
            gpm = production * data["Gold Gain"] * 60
            
            allocs.append({
                "name": name,
                "count": count,
                "tier": data["TierMultiplier"],
                "production": production,
                "gpm": gpm
            })
            results["total_gpm"] += gpm
            
            for res, qty in data["Outputs"].items():
                resource_flows[res] += qty * production
            for res, qty in data["Inputs"].items():
                if res != "Leek":
                    resource_flows[res] -= qty * production

        # Determine achieved tier for GPM
        max_tier = 0
        achieved_tier_name = "None"
        achieved_tier_value = 0
        for alloc in allocs:
            if alloc['tier'] > achieved_tier_value:
                achieved_tier_value = alloc['tier']
                achieved_tier_name = alloc['name']
        results["achieved_tier_name"] = achieved_tier_name
        results["achieved_tier_value"] = achieved_tier_value

        # Filter build order to only include used recipes
        used_names = {alloc['name'] for alloc in allocs}
        filtered_build_order = [name for name in self.build_order if name in used_names]
        results["build_order"] = filtered_build_order

        results["allocations"] = allocs
        results["resource_balance"] = resource_flows
        return results

    def print_results(self, results):
        if not results:
            print("\nNo valid production chain found")
            return
            
        print(f"\n{'='*50}")
        print(f"{results['mode'].upper()} RESULTS")
        print(f"Total GPM: {results['total_gpm']:,.0f}")
        
        if results['mode'] == "GPM":
            print(f"Achieved Tier: {results['achieved_tier_name']} (Tier {results['achieved_tier_value']}x)")
        elif results['mode'] == "Tier":
            print(f"Achieved Tier: {results['target_tier']} (Tier {self.recipes[results['target_tier']]['TierMultiplier']}x)")
            
        print("\nOptimal Factory Chain:")
        for alloc in results["allocations"]:
            if alloc['count'] > 0:
                print(f"  {alloc['name']}: {alloc['count']} factories")
                print(f"    Production: {alloc['production']:.2f}/s")
                print(f"    Tier Value: {alloc['tier']}x")
                print(f"    GPM: {alloc['gpm']:,.0f}")
        
        print("\nDependency-Validated Build Order:")
        for idx, name in enumerate(results["build_order"], 1):
            print(f"  {idx}. {name}")
        
        print("\nResource Balance:")
        for res, flow in results["resource_balance"].items():
            if res != "Leek" and abs(flow) >= 1e-6:
                status = "✅ OVERFLOW" if flow > 0 else "❌ DEFICIT"
                print(f"  {res}: {abs(flow):.2f}/s {status}")
        print("="*50)

# JSON Loading and Execution
if __name__ == "__main__":
    with open('recipes.json') as f:
        recipes = json.load(f, object_pairs_hook=OrderedDict)
    
    optimizer = FactoryOptimizer(
        recipes=recipes,
        rate_multiplier=2,
        factory_level=20,
        total_factories=11
    )
    
    print("=== GPM Optimization ===")
    gpm_results = optimizer.optimize_gpm()
    optimizer.print_results(gpm_results)
    
    print("\n=== Tier Optimization ===")
    tier_results = optimizer.optimize_tiers()
    optimizer.print_results(tier_results)