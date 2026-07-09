import time


def linear_search(arr, target):
    
    for i in range(len(arr)):
        if arr[i] == target:
            return i  
    return -1  

def binary_search(arr, target):
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
            
    return -1  
if __name__ == "__main__":

    sample_list = [10, 23, 45, 70, 11, 15]
    sample_list.sort() 
    
    target_value = 45
    print(f"Dataset (Sorted): {sample_list}")
    print(f"Target Element to find: {target_value}\n")

    # --- Executing Linear Search ---

    print("--- Running Linear Search ---")
    start_time = time.perf_counter()
    linear_res = linear_search(sample_list, target_value)
    linear_time = time.perf_counter() - start_time
    print(f"Element found at index: {linear_res}")
    print(f"Execution Time: {linear_time:.8f} seconds\n")

    # --- Executing Binary Search ---

    print("--- Running Binary Search ---")
    start_time = time.perf_counter()
    binary_res = binary_search(sample_list, target_value)
    binary_time = time.perf_counter() - start_time
    print(f"Element found at index: {binary_res}")
    print(f"Execution Time: {binary_time:.8f} seconds\n")


    print("--- Performance Comparison Summary ---")
    print(f"Linear Search Time: {linear_time:.8f} s")
    print(f"Binary Search Time: {binary_time:.8f} s")
    
    if binary_time <= linear_time:
        print("\nConclusion: Binary Search processed the query faster or equally efficiently.")
