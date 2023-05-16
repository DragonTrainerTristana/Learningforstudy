function alpha = maxsum_update_alpha(alpha, rho, damping, ppg)
    old = alpha;
    new = old;
    for i=1:ppg
        for j=1:ppg
            tmp = rho;
            tmp(i,j) = -inf;
            new(i,j) = -max(tmp(i,:));
        end
    end
    alpha = damping*new + (1-damping)*old;
end
