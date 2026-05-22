import os
import itertools
import numpy as np
from bezier_math import Bezier

class BezierLUTManager:
    """
    Manages computation, disk caching, and memory storage of Bezier transition LUTs.
    """
    def __init__(self, cache_dir="lut_cache"):
        self.cache_dir = cache_dir
        self.memory_cache = {}
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_lut(self, order: int, a: float, b: float, gamma: float, num_pts: int = 256):
        print(f"[DEBUG] Requesting LUT for order={order}, a={a}, b={b}, gamma={gamma}, num_pts={num_pts}")
        sig = (order, round(a, 6), round(b, 6), round(gamma, 6), num_pts)
        if sig in self.memory_cache:
            print("[DEBUG] LUT found in memory cache.")
            return self.memory_cache[sig]

        filename = f"bezier_lut_o{order}_a{a:.4f}_b{b:.4f}_g{gamma:.4f}_n{num_pts}.npz"
        filepath = os.path.join(self.cache_dir, filename)

        if os.path.exists(filepath):
            print(f"[DEBUG] Loading LUT from file: {filepath}")
            data = np.load(filepath)
            lut = (data['x'], data['y'], data['dydx'], data['d2ydx2'])
            self.memory_cache[sig] = lut
            return lut

        else:
            print("[DEBUG] Computing LUT from scratch.")
            # Compute from scratch if not found
            bz = Bezier(order, a, b, gamma)
            t = np.linspace(0.0, 1.0, num_pts)
            
            x = bz._eval(t, bz.Wx).flatten()
            y = bz._eval(t, bz.Wy).flatten()
            
            # Verify strict monotonicity to ensure safe interpolation
            dx_dt = bz._eval1(t, bz.Wx).flatten()
            if not np.all(dx_dt > 1e-12):
                raise ValueError(f"Parameters yield non-monotone mapping: a={a}, b={b}, gamma={gamma}")

            dydx = bz.dydx(t).flatten()
            d2ydx2 = bz.d2ydx2(t).flatten()

            np.savez(filepath, x=x, y=y, dydx=dydx, d2ydx2=d2ydx2)
            lut = (x, y, dydx, d2ydx2)
            self.memory_cache[sig] = lut
            print("[DEBUG] LUT computation complete.")
            return lut

    def precompute_batch(self, orders, a_vals, b_vals, gammas, num_pts=2000):
        """
        Generates and caches LUTs for all combinations of the provided parameter lists.
        """
        print("Pre-computing Bezier LUTs...")
        count = 0
        
        # itertools.product creates every possible combination of the input lists
        for order, a, b, gamma in itertools.product(orders, a_vals, b_vals, gammas):
            try:
                self.get_lut(order, a, b, gamma, num_pts)
                count += 1
                print(f"  [+] Cached: Order={order}, a={a:.4f}, b={b:.4f}, gamma={gamma:.4f}")
            except ValueError as e:
                # Skips parameter combinations that fold back on themselves (not monotone)
                print(f"  [-] Skipped (Invalid Math): Order={order}, a={a:.4f}, b={b:.4f}, gamma={gamma:.4f}")
        
        print(f"Finished pre-computing {count} valid lookup tables.\n")

lut_manager = BezierLUTManager()

# =============================================================================
# Initialization Script
# Run this file directly to pre-generate your tables before optimization starts
# =============================================================================
if __name__ == "__main__":
    # Define the ranges for your parameters here
    orders_to_compute = [3, 5]               # Cubic and Quintic
    h_vals            = [0.05]               # Transition half-width (a)
    h_ext_vals        = [0.15]               # Total width (b = h + extension)
    
    # Generate 5 different gamma values between -0.1 and 0.1
    gammas_to_compute = np.linspace(-0.1, 0.1, 5).tolist() 
    
    # Run the batch generator
    lut_manager.precompute_batch(
        orders_to_compute, 
        h_vals, 
        h_ext_vals, 
        gammas_to_compute
    )