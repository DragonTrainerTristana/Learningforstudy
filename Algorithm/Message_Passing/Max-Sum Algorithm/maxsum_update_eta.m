function alpha = maxsum_update_eta(eta, beta, damping, ppg)
    old = eta;
    new = old;
    for i=1:ppg
        for j=1:ppg
            tmp = beta;
            tmp(i,j) = -inf;
            new(i,j) = -max(tmp(:,j));
        end
    end
    alpha = damping*new + (1-damping)*old;
end
