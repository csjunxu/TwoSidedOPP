function  [im_out, par] = WLSWSC_SigmaNew_1AR(par)
im_out    =   par.nim;
% parameters for noisy image
[h,  w, ch]      =  size(im_out);
par.h = h;
par.w = w;
par.ch = ch;
par = SearchNeighborIndex( par );
%
Sigma = ones(1, par.maxrc);
Wls = ones(1, par.maxrc); % invWls is the inverse of Wls and equals to 1 ./ Wls;
for ite  =  1 : par.outerIter
    %     % iterative regularization
    %     im_out = im_out+par.delta*(par.nim - im_out);
    % image to patches and estimate local noise variance
    Y = Image2PatchNew( im_out, par );
    % estimation of noise variance
    if mod(ite-1, par.innerIter)==0
        par.nlsp = par.nlsp - 10;
        % searching  non-local patches
        blk_arr = Block_Matching( Y, par );
    end
    % Weighted Sparse Coding
    Y_hat = zeros(par.ps2ch, par.maxrc, 'single');
    W_hat = zeros(par.ps2ch, par.maxrc, 'single');
    for i = 1:par.lenrc
        index = blk_arr(:, i);
        nlY = Y( : , index );
        DC = mean(nlY, 2);
        nDCnlY = bsxfun(@minus, nlY, DC);
        if mod(ite-1, par.innerIter)==0
            Sigma(index) = par.lambdaw * sqrt(var(nDCnlY));
            Wls(index) = 1 ./ Sigma(index);
        end
        % weighted least square and weighted sparse coding
        nDCnlYhat = WLSWSCNew(nDCnlY, Wls(index), par);
        % update Weighting matrix for weighted least square
        Wls(index) = par.lambdals ./ sqrt(abs(Sigma(index) - mean((nDCnlY - nDCnlYhat).^2)));
        % add DC components
        nlYhat = bsxfun(@plus, nDCnlYhat, DC);
        % aggregation
        Y_hat(:, index) = Y_hat(:, index) + nlYhat;
        W_hat(:, index) = W_hat(:, index) + ones(par.ps2ch, par.nlsp);
    end
    % Reconstruction
    im_out = PGs2Image(Y_hat, W_hat, par);
    % calculate the PSNR
    PSNR =   csnr( im_out * 255, par.I * 255, 0, 0 );
    SSIM      =  cal_ssim( im_out * 255, par.I * 255, 0, 0 );
    fprintf('Iter %d : PSNR = %2.4f, SSIM = %2.4f\n', ite, PSNR, SSIM);
    par.PSNR(ite, par.image) = PSNR;
    par.SSIM(ite, par.image) = SSIM;
end
return;

