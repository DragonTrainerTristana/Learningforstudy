%%
% 
clear; clc;

% definitions
inf = 10^60;
damping = 1.0; % ratio of the new values % possible GUI_Slider
ppg = 10; %people per group % possibleGUI_Slider
iter_max = ppg*10; % this was about enough number of iterations (heuristically designed)

w = rand(ppg,ppg);

% space allocation
alpha = zeros(ppg,ppg);
rho = zeros(ppg,ppg);
eta = zeros(ppg,ppg);
beta = zeros(ppg,ppg);

% message passing
for iter = 1:iter_max
    beta = maxsum_update_beta(alpha, beta, w, damping);
    rho = maxsum_update_rho(rho, eta, w, damping);
    alpha = maxsum_update_alpha(alpha, rho, damping, ppg);
    eta = maxsum_update_eta(eta, beta, damping, ppg);
end

% match result evaluation
D = eta+alpha+w;
for i=1:ppg
    for j=1:ppg
        if(D(i,j)==max(D(i,:)))
            D(i,:) = 0;
            D(i,j) = 1;
        end
    end
end

disp(D)