function rho = maxsum_update_rho(rho, eta, w, damping)
    old = rho;
    new = w+eta;
    rho = damping*new + (1-damping)*old;
end
