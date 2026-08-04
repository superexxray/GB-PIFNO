import matplotlib.pyplot as plt
import numpy as np
import json
import os
import torch
from eval_metrics import radial_energy_spectrum, enstrophy

def plot_gradient_norm_pathology(static_history_file, gb_history_file):
    """
    Phase 9.3: Gradient norm pathology plot
    Static PIFNO vs. GB-PIFNO gradient norms over training epochs
    """
    if not os.path.exists(static_history_file) or not os.path.exists(gb_history_file):
        print("History files not found for gradient norm pathology plot.")
        return

    with open(static_history_file, 'r') as f:
        static_hist = json.load(f)
    with open(gb_history_file, 'r') as f:
        gb_hist = json.load(f)

    epochs = np.arange(len(gb_hist['g_data']))
    
    plt.figure(figsize=(10, 6))
    
    # GB-PIFNO
    plt.plot(epochs, gb_hist['g_data'], label='GB-PIFNO ||∇L_data||', color='blue', linestyle='-')
    plt.plot(epochs, gb_hist['g_pde'], label='GB-PIFNO ||∇L_pde||', color='cyan', linestyle='--')
    
    # Static PIFNO (assuming we logged it)
    if 'g_data' in static_hist and 'g_pde' in static_hist:
        plt.plot(epochs, static_hist['g_data'], label='Static PIFNO ||∇L_data||', color='red', linestyle='-')
        plt.plot(epochs, static_hist['g_pde'], label='Static PIFNO ||∇L_pde||', color='orange', linestyle='--')

    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm (L2)')
    plt.title('Gradient Norm Pathology: Static vs GB-PIFNO')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig('gradient_norm_pathology.png', dpi=300)
    print("Saved gradient_norm_pathology.png")
    plt.close()

def plot_radial_energy_spectrum(pred_omega, target_omega):
    """
    Phase 9.1: 1D radial energy spectrum plot E(k) vs k (log-log)
    """
    # Pred and target should be (B, C, H, W)
    # We take the mean across batch and channels
    pred_spectrum = radial_energy_spectrum(pred_omega).mean(dim=(0, 1)).cpu().numpy()
    target_spectrum = radial_energy_spectrum(target_omega).mean(dim=(0, 1)).cpu().numpy()
    
    k = np.arange(1, len(pred_spectrum))
    pred_spectrum = pred_spectrum[1:]
    target_spectrum = target_spectrum[1:]
    
    plt.figure(figsize=(8, 6))
    plt.loglog(k, target_spectrum, label='Ground Truth', color='black', linewidth=2)
    plt.loglog(k, pred_spectrum, label='Predicted (GB-PIFNO)', color='blue')
    
    # Kolmogorov k^-3 reference slope
    k_ref = k[1:]
    ref_slope = target_spectrum[1] * (k_ref / k_ref[0])**-3
    plt.loglog(k_ref, ref_slope, label='Kolmogorov $k^{-3}$', color='gray', linestyle='--')
    
    plt.xlabel('Wavenumber $k$')
    plt.ylabel('Energy Spectrum $E(k)$')
    plt.title('1D Radial Energy Spectrum')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig('radial_energy_spectrum.png', dpi=300)
    print("Saved radial_energy_spectrum.png")
    plt.close()

def plot_enstrophy_dissipation(pred_omega, target_omega, dt=1.0):
    """
    Phase 9.2: Enstrophy dissipation plot Omega(t) vs time
    """
    # (B, T, H, W)
    pred_ens = enstrophy(pred_omega).mean(dim=0).cpu().numpy() # (T,)
    target_ens = enstrophy(target_omega).mean(dim=0).cpu().numpy() # (T,)
    
    time = np.arange(len(pred_ens)) * dt
    
    plt.figure(figsize=(8, 6))
    plt.plot(time, target_ens, label='Ground Truth', color='black', linewidth=2)
    plt.plot(time, pred_ens, label='Predicted (GB-PIFNO)', color='blue')
    
    plt.xlabel('Time')
    plt.ylabel('Enstrophy $\Omega(t)$')
    plt.title('Enstrophy Dissipation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('enstrophy_dissipation.png', dpi=300)
    print("Saved enstrophy_dissipation.png")
    plt.close()

def plot_sparsity_degradation(results_json):
    """
    Phase 9.4: Sparsity degradation curves
    """
    if not os.path.exists(results_json):
        print(f"Results file {results_json} not found.")
        return
        
    with open(results_json, 'r') as f:
        results = json.load(f)
        
    # Assume format: { "model_name": { "0.01": l2_error, "0.05": l2_error, ... } }
    plt.figure(figsize=(8, 6))
    for model, sparsities in results.items():
        levels = sorted([float(k) for k in sparsities.keys()])
        errors = [sparsities[str(k)] for k in levels]
        plt.plot(levels, errors, marker='o', label=model)
        
    plt.xscale('log')
    plt.xlabel('Sparsity Level (Fraction of observed pixels)')
    plt.ylabel('Relative $L_2$ Error')
    plt.title('Sparsity Degradation Curves')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig('sparsity_degradation.png', dpi=300)
    print("Saved sparsity_degradation.png")
    plt.close()

if __name__ == '__main__':
    print("Generating Gradient Norm Pathology plot from saved logs...")
    plot_gradient_norm_pathology('static_history.json', 'gb_history.json')
    print("Done! Check your folder for gradient_norm_pathology.png.")
