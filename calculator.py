import pulp
import json
from collections import OrderedDict

class FactoryOptimizer:
    def __init__(self, recipes, rate_multiplier, factory_level, total_factories):
        self.recipes = OrderedDict(sorted(recipes.items(), 
            key=lambda x: -x[1]["TierMultiplier"]))
        self.rate_multiplier = rate_multiplier
        self.factory_level = factory_level
        self.total_factories = total_factories
        self._clean_recipes()
        self._calculate_rates()

    def _clean_recipes(self):
        """Remove inefficient conversions and invalid recipes"""
        self.recipes.pop("Barrel to Leek", None)
        
    def _calculate_rates(self):
        """Pre-calculate all production metrics"""
        for name, data in self.recipes.items():
            effective_rate = data["Base Rate"] * (self.rate_multiplier ** (self.factory_level - 1))
            self.recipes[name].update({
                "Effective Rate": effective_rate,
                "GPM": effective_rate * data["Gold Gain"] * 60
            })

    def _solve_problem(self, prob):
        """Silent solver with error handling"""
        try:
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            return pulp.LpStatus[prob.status] == "Optimal"
        except Exception as e:
            print(f"Solving error: {str(e)}")
            return False

    def optimize_gpm(self):
        """Maximize gold production"""
        prob = pulp.LpProblem("GPM_Optimization", pulp.LpMaximize)
        vars = pulp.LpVariable.dicts("F", self.recipes.keys(), lowBound=0, cat="Integer")
        
        # Objective: Total GPM
        prob += pulp.lpSum(vars[name] * self.recipes[name]["GPM"] for name in self.recipes)
        prob += pulp.lpSum(vars.values()) <= self.total_factories
        
        # Resource constraints
        resources = set()
        for recipe in self.recipes.values():
            resources.update(recipe["Inputs"].keys())
            resources.update(recipe["Outputs"].keys())
        resources.discard("Leek")
        
        for res in resources:
            prob += pulp.lpSum(
                vars[name] * qty * self.recipes[name]["Effective Rate"]
                for name in self.recipes
                for output, qty in self.recipes[name]["Outputs"].items() if output == res
            ) >= pulp.lpSum(
                vars[name] * qty * self.recipes[name]["Effective Rate"]
                for name in self.recipes
                for input_, qty in self.recipes[name]["Inputs"].items() if input_ == res
            )
        
        if not self._solve_problem(prob):
            return None
            
        return self._parse_results(vars, "GPM")

    def optimize_tiers(self):
        """Maximize highest achievable tier"""
        for tier_name in self.recipes:
            prob = pulp.LpProblem("Tier_Optimization", pulp.LpMaximize)
            vars = pulp.LpVariable.dicts("F", self.recipes.keys(), lowBound=0, cat="Integer")
            
            # Objective: Maximize target tier production
            prob += vars[tier_name] * self.recipes[tier_name]["Effective Rate"]
            prob += pulp.lpSum(vars.values()) <= self.total_factories
            
            # Resource constraints
            resources = set()
            for recipe in self.recipes.values():
                resources.update(recipe["Inputs"].keys())
                resources.update(recipe["Outputs"].keys())
            resources.discard("Leek")
            
            for res in resources:
                prob += pulp.lpSum(
                    vars[name] * qty * self.recipes[name]["Effective Rate"]
                    for name in self.recipes
                    for output, qty in self.recipes[name]["Outputs"].items() if output == res
                ) >= pulp.lpSum(
                    vars[name] * qty * self.recipes[name]["Effective Rate"]
                    for name in self.recipes
                    for input_, qty in self.recipes[name]["Inputs"].items() if input_ == res
                )
            
            if self._solve_problem(prob) and vars[tier_name].varValue > 0:
                return self._parse_results(vars, "Tier", tier_name)
        
        return None

    def _parse_results(self, vars, mode, target_tier=None):
        """Format results consistently"""
        results = []
        total_gpm = 0
        max_tier = 0
        
        for name in self.recipes:
            count = int(vars[name].varValue)
            if count <= 0:
                continue
                
            data = self.recipes[name]
            gpm = data["GPM"] * count
            results.append({
                "name": name,
                "count": count,
                "tier": data["TierMultiplier"],
                "gpm": gpm,
                "rate": data["Effective Rate"] * count
            })
            total_gpm += gpm
            max_tier = max(max_tier, data["TierMultiplier"])
        
        return {
            "total_gpm": total_gpm,
            "max_tier": max_tier,
            "target_tier": target_tier,
            "allocations": sorted(results, key=lambda x: (-x["tier"], -x["gpm"])),
            "mode": mode
        }

    def print_results(self, results):
        """Clean formatted output"""
        if not results:
            print("No feasible solution found")
            return
            
        print(f"\n{'='*50}")
        print(f"{'GPM Optimization' if results['mode'] == 'GPM' else 'Tier Optimization'} Results")
        print(f"{'-'*50}")
        print(f"Total GPM: {results['total_gpm']:,.0f}")
        print(f"Highest Tier: {results['max_tier']}x")
        if results['mode'] == 'Tier':
            print(f"Target Tier Achieved: {results['target_tier']}")
            
        print("\nFactory Allocation:")
        for alloc in results["allocations"]:
            print(f"  {alloc['name']}:")
            print(f"    Factories: {alloc['count']}")
            print(f"    Tier: {alloc['tier']}x")
            print(f"    GPM: {alloc['gpm']:,.0f}")
            print(f"    Production: {alloc['rate']:.6f}/s")
        print("="*50)

# Usage example
if __name__ == "__main__":
    # Load recipes (replace with JSON loading if needed)
    with open('recipes.json') as f:
        recipes = json.load(f, object_pairs_hook=OrderedDict)
    
    # Initialize optimizer
    optimizer = FactoryOptimizer(
        recipes=recipes,
        rate_multiplier=2,
        factory_level=11,
        total_factories=8
    )
    
    # Compare both strategies
    print("\n=== Running GPM Optimization ===")
    gpm_results = optimizer.optimize_gpm()
    optimizer.print_results(gpm_results)
    
    print("\n=== Running Tier Optimization ===")
    tier_results = optimizer.optimize_tiers()
    optimizer.print_results(tier_results)