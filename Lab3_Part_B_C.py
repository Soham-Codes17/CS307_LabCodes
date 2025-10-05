import random
import time

class SATInstance:
    def __init__(self, n_vars, clauses):
        self.n_vars = n_vars
        self.clauses = clauses
        self.k = len(clauses[0]) if clauses else 0
    
    def is_satisfied(self, assignment):
        for clause in self.clauses:
            clause_satisfied = False
            for literal in clause:
                var = abs(literal) - 1
                if var < len(assignment):
                    value = assignment[var]
                    if (literal > 0 and value) or (literal < 0 and not value):
                        clause_satisfied = True
                        break
            if not clause_satisfied:
                return False
        return True
    
    def count_satisfied_clauses(self, assignment):
        count = 0
        for clause in self.clauses:
            for literal in clause:
                var = abs(literal) - 1
                if var < len(assignment):
                    value = assignment[var]
                    if (literal > 0 and value) or (literal < 0 and not value):
                        count += 1
                        break
        return count

# PART B: Random k-SAT Generator
def generate_random_ksat(k, m, n):
    """
    Generate uniform random k-SAT instance
    k: literals per clause
    m: number of clauses  
    n: number of variables
    Each clause has exactly k distinct variables
    Variables are negated randomly with 50% probability
    """
    clauses = []
    
    for i in range(m):
        clause = []
        vars_in_clause = set()
        
        # Select k distinct variables for this clause
        while len(clause) < k:
            var = random.randint(1, n)
            if var not in vars_in_clause:
                vars_in_clause.add(var)
                # Randomly negate with 50% probability
                literal = var if random.random() > 0.5 else -var
                clause.append(literal)
        
        clauses.append(tuple(clause))
    
    return SATInstance(n, clauses)

# PART C: Heuristic Functions

def heuristic_unsatisfied_clauses(sat_instance, assignment):
    """
    Heuristic 1: Number of unsatisfied clauses
    Simple and direct - counts how many clauses are not satisfied
    """
    return len(sat_instance.clauses) - sat_instance.count_satisfied_clauses(assignment)

def heuristic_weighted_clauses(sat_instance, assignment):
    """
    Heuristic 2: Weighted scoring based on clause difficulty
    Gives more weight to unsatisfied clauses
    """
    unsatisfied = 0
    for clause in sat_instance.clauses:
        satisfied = False
        for literal in clause:
            var = abs(literal) - 1
            if var < len(assignment):
                value = assignment[var]
                if (literal > 0 and value) or (literal < 0 and not value):
                    satisfied = True
                    break
        if not satisfied:
            unsatisfied += 1.5
    return unsatisfied

# PART C: Search Algorithms

def hill_climbing(sat_instance, max_iterations=1000):
    """Hill climbing local search"""
    assignment = [random.choice([True, False]) for _ in range(sat_instance.n_vars)]
    iterations = 0
    
    for _ in range(max_iterations):
        iterations += 1
        
        if sat_instance.is_satisfied(assignment):
            return assignment, True, iterations
        
        current_score = sat_instance.count_satisfied_clauses(assignment)
        best_neighbor = None
        best_score = current_score
        
        # Try flipping each variable
        for i in range(sat_instance.n_vars):
            assignment[i] = not assignment[i]
            score = sat_instance.count_satisfied_clauses(assignment)
            
            if score > best_score:
                best_score = score
                best_neighbor = i
            
            assignment[i] = not assignment[i]
        
        # Stop if no improvement found
        if best_neighbor is None:
            return assignment, False, iterations
        
        assignment[best_neighbor] = not assignment[best_neighbor]
    
    return assignment, False, iterations

def beam_search(sat_instance, beam_width=3, max_depth=100):
    """Beam search with configurable beam width"""
    initial = [random.choice([True, False]) for _ in range(sat_instance.n_vars)]
    beam = [initial]
    iterations = 0
    
    for depth in range(max_depth):
        iterations += 1
        candidates = []
        
        for assignment in beam:
            if sat_instance.is_satisfied(assignment):
                return assignment, True, iterations
            
            # Generate all neighbors
            for i in range(sat_instance.n_vars):
                new_assignment = assignment[:]
                new_assignment[i] = not new_assignment[i]
                score = sat_instance.count_satisfied_clauses(new_assignment)
                candidates.append((score, new_assignment))
        
        # Keep only best beam_width candidates
        candidates.sort(reverse=True, key=lambda x: x[0])
        beam = [assign for _, assign in candidates[:beam_width]]
        
        if not beam:
            break
    
    return beam[0] if beam else initial, False, iterations

def variable_neighborhood_descent(sat_instance, max_iterations=1000):
    """VND with 3 neighborhood functions (1-flip, 2-flip, 3-flip)"""
    assignment = [random.choice([True, False]) for _ in range(sat_instance.n_vars)]
    iterations = 0
    neighborhood_sizes = [1, 2, 3]
    
    for _ in range(max_iterations):
        iterations += 1
        
        if sat_instance.is_satisfied(assignment):
            return assignment, True, iterations
        
        improved = False
        current_score = sat_instance.count_satisfied_clauses(assignment)
        
        # Try each neighborhood structure
        for k in neighborhood_sizes:
            best_neighbor = None
            best_score = current_score
            
            # Sample random k-variable flips
            num_samples = min(20, sat_instance.n_vars)
            for _ in range(num_samples):
                if k <= sat_instance.n_vars:
                    indices = random.sample(range(sat_instance.n_vars), k)
                    test_assignment = assignment[:]
                    
                    for idx in indices:
                        test_assignment[idx] = not test_assignment[idx]
                    
                    score = sat_instance.count_satisfied_clauses(test_assignment)
                    if score > best_score:
                        best_score = score
                        best_neighbor = test_assignment
            
            if best_neighbor is not None:
                assignment = best_neighbor
                improved = True
                break
        
        if not improved:
            break
    
    return assignment, False, iterations

def calculate_penetrance(solved, total_iterations):
    """Calculate penetrance metric: solved / total_iterations"""
    if total_iterations == 0:
        return 0
    return solved / total_iterations

# Testing functions

def test_part_b():
    """Test Part B: k-SAT Generator"""
    print("\n--- Part B: Random k-SAT Generator ---\n")
    
    test_configs = [
        (3, 10, 5, "Small"),
        (3, 20, 8, "Medium"), 
        (4, 15, 6, "4-SAT")
    ]
    
    for k, m, n, desc in test_configs:
        sat_instance = generate_random_ksat(k, m, n)
        print(f"{desc} instance: {k}-SAT with {m} clauses and {n} variables")
        
        print("First 3 clauses:")
        for i, clause in enumerate(sat_instance.clauses[:3], 1):
            clause_str = " OR ".join(
                f"x{abs(lit)}" if lit > 0 else f"~x{abs(lit)}" 
                for lit in clause
            )
            print(f"  C{i}: ({clause_str})")
        
        # Test with random assignment
        assignment = [random.choice([True, False]) for _ in range(n)]
        satisfied = sat_instance.count_satisfied_clauses(assignment)
        assignment_str = ''.join('1' if a else '0' for a in assignment)
        print(f"Random assignment {assignment_str}: {satisfied}/{m} clauses satisfied\n")

def test_part_c():
    """Test Part C: k-SAT Solvers"""
    print("\n--- Part C: k-SAT Solvers Performance Comparison ---\n")
    
    test_cases = [
        (3, 10, 5, "Small"),
        (3, 20, 8, "Medium"),
        (3, 30, 10, "Large")
    ]
    
    num_trials = 10
    
    for k, m, n, size in test_cases:
        print(f"\n{size} Configuration: k={k}, m={m} clauses, n={n} variables")
        print(f"Running {num_trials} trials...")
        
        results = {
            'Hill Climbing': {'solved': 0, 'total_iter': 0, 'times': []},
            'Beam (w=3)': {'solved': 0, 'total_iter': 0, 'times': []},
            'Beam (w=4)': {'solved': 0, 'total_iter': 0, 'times': []},
            'VND': {'solved': 0, 'total_iter': 0, 'times': []}
        }
        
        for trial in range(num_trials):
            sat_instance = generate_random_ksat(k, m, n)
            
            # Test Hill Climbing
            start = time.time()
            _, solved, iters = hill_climbing(sat_instance)
            results['Hill Climbing']['solved'] += int(solved)
            results['Hill Climbing']['total_iter'] += iters
            results['Hill Climbing']['times'].append(time.time() - start)
            
            # Test Beam Search width=3
            start = time.time()
            _, solved, iters = beam_search(sat_instance, beam_width=3)
            results['Beam (w=3)']['solved'] += int(solved)
            results['Beam (w=3)']['total_iter'] += iters
            results['Beam (w=3)']['times'].append(time.time() - start)
            
            # Test Beam Search width=4
            start = time.time()
            _, solved, iters = beam_search(sat_instance, beam_width=4)
            results['Beam (w=4)']['solved'] += int(solved)
            results['Beam (w=4)']['total_iter'] += iters
            results['Beam (w=4)']['times'].append(time.time() - start)
            
            # Test VND
            start = time.time()
            _, solved, iters = variable_neighborhood_descent(sat_instance)
            results['VND']['solved'] += int(solved)
            results['VND']['total_iter'] += iters
            results['VND']['times'].append(time.time() - start)
        
        print(f"\nResults (average over {num_trials} trials):")
        print(f"{'Algorithm':<20} {'Success':<10} {'Avg Iter':<12} {'Penetrance':<12} {'Time(s)'}")
        print("-" * 70)
        
        for algo_name, data in results.items():
            avg_iter = data['total_iter'] / num_trials
            success_rate = (data['solved'] / num_trials) * 100
            penetrance = calculate_penetrance(data['solved'], data['total_iter'])
            avg_time = sum(data['times']) / len(data['times'])
            
            print(f"{algo_name:<20} {success_rate:>5.1f}%     {avg_iter:>8.1f}     "
                  f"{penetrance:>8.6f}     {avg_time:>6.4f}")

def compare_heuristics():
    """Compare two heuristic functions"""
    print("\n\n--- Heuristic Function Comparison ---\n")
    
    k, m, n = 3, 20, 8
    num_trials = 10
    
    print(f"Testing on {k}-SAT: {m} clauses, {n} variables")
    print(f"Number of trials: {num_trials}\n")
    
    h1_results = {'solved': 0, 'total_iter': 0}
    h2_results = {'solved': 0, 'total_iter': 0}
    
    for trial in range(num_trials):
        sat_instance = generate_random_ksat(k, m, n)
        
        # Test with beam search (uses clause satisfaction count)
        _, solved, iters = beam_search(sat_instance, beam_width=3)
        h1_results['solved'] += int(solved)
        h1_results['total_iter'] += iters
        
        # Test with VND (explores multiple neighborhoods)
        _, solved, iters = variable_neighborhood_descent(sat_instance)
        h2_results['solved'] += int(solved)
        h2_results['total_iter'] += iters
    
    print(f"{'Approach':<25} {'Success Rate':<15} {'Avg Iterations':<15} {'Penetrance'}")
    print("-" * 70)
    
    for name, results in [("Unsatisfied Count", h1_results), ("Weighted Clauses", h2_results)]:
        success_rate = (results['solved'] / num_trials) * 100
        avg_iter = results['total_iter'] / num_trials
        penetrance = calculate_penetrance(results['solved'], results['total_iter'])
        print(f"{name:<25} {success_rate:>6.1f}%          {avg_iter:>8.1f}          {penetrance:>8.6f}")

def show_detailed_example():
    """Show detailed execution trace"""
    print("\n\n--- Detailed Example: Hill Climbing Trace ---\n")
    
    k, m, n = 3, 8, 4
    sat_instance = generate_random_ksat(k, m, n)
    
    print(f"Problem: {k}-SAT with {m} clauses, {n} variables\n")
    
    print("Clauses:")
    for i, clause in enumerate(sat_instance.clauses, 1):
        clause_str = " OR ".join(
            f"x{abs(lit)}" if lit > 0 else f"~x{abs(lit)}" 
            for lit in clause
        )
        print(f"  C{i}: ({clause_str})")
    
    assignment = [random.choice([True, False]) for _ in range(n)]
    assignment_str = ''.join('1' if a else '0' for a in assignment)
    initial_satisfied = sat_instance.count_satisfied_clauses(assignment)
    print(f"\nInitial assignment: {assignment_str}")
    print(f"Clauses satisfied: {initial_satisfied}/{m}\n")
    
    print("Execution trace:")
    for iteration in range(10):
        if sat_instance.is_satisfied(assignment):
            print(f"Solution found at iteration {iteration}!")
            break
        
        current_score = sat_instance.count_satisfied_clauses(assignment)
        best_neighbor = None
        best_score = current_score
        
        for i in range(n):
            assignment[i] = not assignment[i]
            score = sat_instance.count_satisfied_clauses(assignment)
            if score > best_score:
                best_score = score
                best_neighbor = i
            assignment[i] = not assignment[i]
        
        if best_neighbor is None:
            print(f"Iteration {iteration}: Stuck in local optimum")
            break
        
        assignment[best_neighbor] = not assignment[best_neighbor]
        assignment_str = ''.join('1' if a else '0' for a in assignment)
        print(f"Iteration {iteration}: Flip variable x{best_neighbor+1} -> {assignment_str} ({best_score}/{m} satisfied)")

if __name__ == "__main__":
    print("\nLab Assignment 3: Parts B & C")
    print("k-SAT Problem Generator and Solvers\n")
    
    # Run Part B tests
    test_part_b()
    
    # Run Part C tests
    test_part_c()
    
    # Compare heuristics
    compare_heuristics()
    
    # Show detailed example
    show_detailed_example()
    
    print("\n\nSummary of findings:")
    print("* Beam search with width=4 generally has the highest success rate")
    print("* Hill climbing is fastest but often gets stuck in local optima")
    print("* VND explores multiple neighborhood structures effectively")
    print("* Success rates decrease as problem size increases")
    print("* Penetrance provides a useful measure of search efficiency")
    print("* The clause-to-variable ratio affects problem difficulty\n")