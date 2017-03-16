% double weighted: weighted least square and weighted sparse coding framework
function  X = LSWSC(Y, par)
% initialize D and S
[U, S, V] = svd(Y * Y', 'econ');
D = V * U';
S = diag(S);
% fixed W for weighted sparse coding
Wsc = repmat(par.lambdasc ./ (S + eps ), [1 size(Y, 2)]); % sqrt(S) ?

f_curr = 0;
for i=1:par.WWIter
    f_prev = f_curr;
    % update C by soft thresholding
    B = D' * Y;
    C = sign(B) .* max(abs(B) - Wsc, 0);
    % update D
    [U, ~, V] = svd( C * Y', 'econ');
    D = V * U';
    
    % energy function
    DT = Y - D * C;
    DT = norm(DT, 'fro');
    %     DT = DT(:)'*DT(:);
    RT = Wsc .*  C;
    RT = sum(sum(abs(RT)));
    f_curr = 0.5 * DT ^ 2 + par.lambdasc * RT;
    %     fprintf('WLSWSC Energy, %d th: %2.8f\n', i, f_curr);
    if (abs(f_prev - f_curr) / f_curr < par.epsilon)
        break;
    end
end
% update X
X = D * C;
return;
