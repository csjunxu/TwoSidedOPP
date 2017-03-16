% double weighted: weighted least square and weighted sparse coding framework
function  X = LSSC(Y, par)
% initialize D as identity matrix
D = eye(size(Y, 1));
f_curr = 0;
for i=1:par.WWIter
    f_prev = f_curr;
    % update C by soft thresholding
    B = D' * Y;
    C = sign(B) .* max(abs(B) - par.lambdasc * par.nSig^2, 0);
%     C = sign(B) .* max(abs(B) - par.lambdasc, 0);
    % update D
    [U, ~, V] = svd( C * Y', 'econ');
    D = V * U';
    % energy function
    DT = Y - D * C;
    DT = norm(DT, 'fro');
    RT = sum(sum(abs(C)));
    f_curr = 0.5 * DT^2 + par.lambdasc * RT;
%     fprintf('LSSC Energy, %d th: %2.8f\n', i, f_curr);
    if (abs(f_prev - f_curr) / f_curr < par.epsilon)
        break;
    end
end
% update X
X = D * C;
return;
