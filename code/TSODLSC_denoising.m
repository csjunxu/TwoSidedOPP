function  [im_out, par]    =   TSODLSC_denoising(par)
im_out    =   par.nim;
% parameters for noisy image
[h,  w, ch]      =  size(im_out);
par.h = h;
par.w = w;
par.ch = ch;
par = SearchNeighborIndex( par );
NY = Image2PatchNew( par.nim, par );
for ite  =  1 : par.outerIter
    % iterative regularization
    im_out = im_out + par.delta * (par.nim - im_out);
    % image to patches and estimate local noise variance
    Y = Image2PatchNew( im_out, par);
    % estimate local noise variance, par.lambdals is put here since the MAP
    % and Bayesian rules
    Sigma = par.lambda*sqrt(abs(repmat(par.nSig^2, 1, size(Y, 2)) - mean((NY - Y).^2))); %Estimated Local Noise Level
    % estimation of noise variance
    if mod(ite-1,par.innerIter)==0
        % par.nlsp = par.nlsp - 10;
        % par.nlsp = min(par.nlsp, 30);
        % searching  non-local patches
        blk_arr = Block_Matching( Y, par);
        if ite == 1
            Sigma = par.nSig * ones(size(Sigma));
        end
    end
    % Weighted Sparse Coding
    Y_hat = zeros(par.ps2ch, par.maxrc, 'single');
    W_hat = zeros(par.ps2ch, par.maxrc, 'single');
    for i = 1:par.lenrc
        index = blk_arr(:, i);
        nlY = Y( : , index );
        DC = mean(nlY, 2);
        par.Sigma = Sigma(index(1));
        nDCnlY = bsxfun(@minus, nlY, DC);
        % Recovered Estimated Patches by weighted least square and weighted
        % sparse coding model
        nDCnlYhat = TSODLSC(nDCnlY, par);
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

