# Clipping diagnostics tracking
clip_count_A = 0
clip_count_D_rev = 0
clip_count_D_irr = 0
clip_count_B = 0

# (中略：状態更新後)
# --- 2. Numerical stabilization clipping applied AFTER safety checks ---
A_clipped = np.clip(A_next, -50.0, 50.0)
D_rev_clipped = np.clip(D_rev_next, 0.0, 200.0)
D_irr_clipped = np.clip(D_irr_next, 0.0, 200.0)
B_clipped = np.clip(B_next, 0.0, 500.0)

clip_count_A += np.sum(A_next != A_clipped)
clip_count_D_rev += np.sum(D_rev_next != D_rev_clipped)
clip_count_D_irr += np.sum(D_irr_next != D_irr_clipped)
clip_count_B += np.sum(B_next != B_clipped)

A, D_rev, D_irr, B = A_clipped, D_rev_clipped, D_irr_clipped, B_clipped
