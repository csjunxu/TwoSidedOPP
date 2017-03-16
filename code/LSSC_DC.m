% single weighted: weighted least square and sparse coding framework
function  X = LSSC_DC(Y, par)
% initialize D
[U, ~, V] = svd(Y * Y', 'econ');
D = V * U';
f_curr = 0;
% update W for weighted sparse coding
Wsc = par.lambdasc / par.Wls^2;
for i=1:par.WWIter
    f_prev = f_curr;
    % update C by soft thresholding
    B = D' * Y;
    C = sign(B) .* max(abs(B) - repmat(Wsc, size(B)), 0);
    % update D
    [U, ~, V] = svd( C * Y', 'econ');
    D = V * U';
    % energy function
    DT = bsxfun(@times, Y - D * C, repmat(par.Wls, [1, size(Y ,2)]));
    DT = norm(DT, 'fro');
    %     DT = DT(:)'*DT(:);
    RT = sum(sum(abs(C)));
    f_curr = 0.5 * DT ^ 2 + par.lambdasc * RT;
    %     fprintf('WLSSC Energy, %d th: %2.8f\n', i, f_curr);
    if (abs(f_prev - f_curr) / f_curr < par.epsilon)
        break;
    end
end
% update X
X = D * C;
return;
