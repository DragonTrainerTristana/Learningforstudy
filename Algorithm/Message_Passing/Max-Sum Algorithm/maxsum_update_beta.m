function beta = maxsum_update_beta(alpha, beta, w, damping)
    old = beta;
    new = w+alpha;
    beta = damping*new + (1-damping)*old;
end
