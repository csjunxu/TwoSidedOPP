% two sided orthogonal dictionary learning with sparse coding
function  X = TSODLSC(Y, par)
% initialize D as identity matrix
T = eye(size(Y, 1));
S = eye(size(Y, 2));
f_curr = 0;
for i=1:par.Iter
    f_prev = f_curr;
    
    % update A by soft thresholding
    B = T' * Y * S';
    A = sign(B) .* max(abs(B) - par.nSig^2, 0);
    
    % update T and S
    [U1, ~, V1] = svd( Y, 'econ');
    [U2, ~, V2] = svd( A, 'econ');
    T = U1 * U2';
    S = V2 * V1';
    
    % energy function
    DT = Y - T * A * S;
    DT = norm(DT, 'fro');
    RT = sum(sum(abs(A)));
    f_curr = 0.5 * DT^2 + par.nSig^2 * RT;
    fprintf('TSODLSC Energy, %d th: %2.8f\n', i, f_curr);
    if (abs(f_prev - f_curr) / f_curr < par.epsilon)
        break;
    end
end
% update X
X = T * A * S;
return;
