import numpy as np
from math import exp, factorial
import time

# ------------------------------
# Problem 2 parameters
# ------------------------------
MAX_BIKES = 20
MAX_MOVE = 5
RENTAL_REWARD = 10
MOVE_COST = 2
DISCOUNT = 0.9

# Poisson means
RENTAL_REQUEST_MEAN = [3, 4]
RETURN_MEAN = [3, 2]

# Precompute Poisson probabilities
MAX_POISSON = 11
POISSON_CACHE = {}

def poisson_prob(n, lam):
    key = (n, lam)
    if key not in POISSON_CACHE:
        POISSON_CACHE[key] = exp(-lam) * (lam ** n) / factorial(n)
    return POISSON_CACHE[key]

# Precompute all probability arrays
req_probs1 = np.array([poisson_prob(i, RENTAL_REQUEST_MEAN[0]) for i in range(MAX_POISSON)])
req_probs2 = np.array([poisson_prob(i, RENTAL_REQUEST_MEAN[1]) for i in range(MAX_POISSON)])
ret_probs1 = np.array([poisson_prob(i, RETURN_MEAN[0]) for i in range(MAX_POISSON)])
ret_probs2 = np.array([poisson_prob(i, RETURN_MEAN[1]) for i in range(MAX_POISSON)])

def expected_return(state, action, value):
    """
    Calculate expected return for a state-action pair
    """
    bikes1, bikes2 = state
    
    # Apply movement with bounds checking
    move_from_1_to_2 = action
    bikes1_after_move = max(0, min(bikes1 - move_from_1_to_2, MAX_BIKES))
    bikes2_after_move = max(0, min(bikes2 + move_from_1_to_2, MAX_BIKES))
    
    # Moving cost
    total_reward = -MOVE_COST * abs(action)
    
    # Expected value calculation
    expected_value = 0.0
    
    for req1 in range(MAX_POISSON):
        p_req1 = req_probs1[req1]
        for req2 in range(MAX_POISSON):
            p_req2 = req_probs2[req2]
            prob_req = p_req1 * p_req2
            
            # Actual rentals possible
            actual_rent1 = min(req1, bikes1_after_move)
            actual_rent2 = min(req2, bikes2_after_move)
            
            # Rental income
            rental_income = (actual_rent1 + actual_rent2) * RENTAL_REWARD
            
            # Remaining bikes after rentals
            remaining1 = bikes1_after_move - actual_rent1
            remaining2 = bikes2_after_move - actual_rent2
            
            # Expected future value over returns
            future_value = 0.0
            for ret1 in range(MAX_POISSON):
                p_ret1 = ret_probs1[ret1]
                for ret2 in range(MAX_POISSON):
                    p_ret2 = ret_probs2[ret2]
                    prob_ret = p_ret1 * p_ret2
                    
                    # New state after returns
                    new_bikes1 = min(remaining1 + ret1, MAX_BIKES)
                    new_bikes2 = min(remaining2 + ret2, MAX_BIKES)
                    
                    future_value += prob_ret * value[new_bikes1, new_bikes2]
            
            expected_value += prob_req * (rental_income + DISCOUNT * future_value)
    
    return total_reward + expected_value

def policy_iteration():
    """
    Policy iteration algorithm for Gbike problem
    """
    # Initialize value function and policy
    value = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))
    policy = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1), dtype=int)
    
    # All possible actions (-5 to +5)
    actions = list(range(-MAX_MOVE, MAX_MOVE + 1))
    
    stable = False
    iteration = 0
    
    print("Starting Policy Iteration for Problem 2...")
    
    while not stable:
        iteration += 1
        start_time = time.time()
        print(f"\nPolicy Iteration {iteration}...")
        
        # Policy Evaluation
        eval_iter = 0
        while True:
            eval_iter += 1
            delta = 0
            new_value = value.copy()
            
            for i in range(MAX_BIKES + 1):
                for j in range(MAX_BIKES + 1):
                    old_val = value[i, j]
                    action = policy[i, j]
                    new_value[i, j] = expected_return((i, j), action, value)
                    delta = max(delta, abs(new_value[i, j] - value[i, j]))
            
            value = new_value
            if delta < 1e-4:
                break
        
        # Policy Improvement
        stable = True
        changes = 0
        
        for i in range(MAX_BIKES + 1):
            for j in range(MAX_BIKES + 1):
                old_action = policy[i, j]
                best_value = -np.inf
                best_action = old_action
                
                for action in actions:
                    # Check if action is valid
                    if action >= 0:  # Moving from loc1 to loc2
                        if action <= min(i, MAX_MOVE) and (j + action) <= MAX_BIKES:
                            action_val = expected_return((i, j), action, value)
                            if action_val > best_value:
                                best_value = action_val
                                best_action = action
                    else:  # Moving from loc2 to loc1
                        if -action <= min(j, MAX_MOVE) and (i - action) <= MAX_BIKES:
                            action_val = expected_return((i, j), action, value)
                            if action_val > best_value:
                                best_value = action_val
                                best_action = action
                
                policy[i, j] = best_action
                if old_action != best_action:
                    stable = False
                    changes += 1
        
        iter_time = time.time() - start_time
        print(f"  Changes: {changes}, Time: {iter_time:.2f}s, Stable: {stable}")
        
        if stable:
            print(f"\nConverged after {iteration} iterations!")
    
    return policy, value

def analyze_policy(policy, value):
    """
    Analyze the optimal policy
    """
    print("\n" + "="*50)
    print("PROBLEM 2 POLICY ANALYSIS")
    print("="*50)
    
    print(f"\nPolicy Statistics:")
    print(f"Min action: {np.min(policy)}")
    print(f"Max action: {np.max(policy)}")
    print(f"Mean action: {np.mean(policy):.2f}")
    
    # Count move directions
    positive_moves = np.sum(policy > 0)
    negative_moves = np.sum(policy < 0)
    zero_moves = np.sum(policy == 0)
    
    print(f"\nMove Directions:")
    print(f"Location 1 → Location 2: {positive_moves} states")
    print(f"Location 2 → Location 1: {negative_moves} states")
    print(f"No movement: {zero_moves} states")
    
    print(f"\nValue Function range: [{np.min(value):.2f}, {np.max(value):.2f}]")

# ------------------------------
# Main execution
# ------------------------------
if __name__ == "__main__":
    start_time = time.time()
    
    policy, value = policy_iteration()
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    
    analyze_policy(policy, value)
    
    print("\nOptimal Policy Matrix:")
    print("Rows: bikes at location 1, Columns: bikes at location 2")
    print(policy)
