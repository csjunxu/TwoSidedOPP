% single weighted: weighted least square and sparse coding framework
function  X = WLSSC_DC(Y, par)
% initialize D
[U, S, V] = svd(Y * Y', 'econ');
D = V * U';
S = diag(S);
% update W for weighted sparse coding
Wsc = par.lambdasc * par.nSig^2 ./ sqrt(S) ;
f_curr = 0;
for i=1:par.WWIter
    f_prev = f_curr;
    % update C by soft thresholding
    B = D' * Y;
    C = sign(B) .* max(abs(B) - repmat(Wsc, [1, size(B, 2)]), 0);
    % update D
    if par.model == 1
        % model 1
        [U, ~, V] = svd( C * Y', 'econ');
    else
        % model 2
        [U, ~, V] = svd( C * Y', 'econ');
    end
    D = V * U';
    % energy function
    DT = Y - D * C;
    DT = norm(DT, 'fro');
    %     DT = DT(:)'*DT(:);
    RT = sum(sum(abs(C)));
    f_curr = 0.5 * DT ^ 2 + par.lambdasc * RT;
%     fprintf('WLSSC Energy, %d th: %2.8f\n', i, f_curr);
    if (abs(f_prev - f_curr) / f_curr < par.epsilon) || f_curr < par.epsilon
        break;
    end
end
% update X
X = D * C;
return;
