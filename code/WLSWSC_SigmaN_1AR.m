function  [im_out, par] = WLSWSC_SigmaN_1AR(par)
im_out    =   par.nim;
% parameters for noisy image
[h,  w, ch]      =  size(im_out);
par.h = h;
par.w = w;
par.ch = ch;
par = SearchNeighborIndex( par );
% original noisy image to patches
NY = Image2PatchNew( par.nim, par );
for ite  =  1 : par.outerIter
    %     % iterative regularization
    %     im_out = im_out+par.delta*(par.nim - im_out);
    % image to patches and estimate local noise variance
    Y = Image2PatchNew( im_out, par );
    % estimate local noise variance, par.lambdals is put here since the MAP
    % and Bayesian rules
    Sigma = par.lambdals * sqrt(abs(repmat(par.nSig0^2, 1, size(Y,2)) - mean((NY - Y).^2))); %Estimated Local Noise Level
    % estimation of noise variance
    if mod(ite-1, par.innerIter)==0
        par.nlsp = par.nlsp - 10;
        % searching  non-local patches
        blk_arr = Block_Matching( Y, par );
        if ite == 1
            Sigma = par.nSig0 * ones(size(Sigma));
        end
    end
    % Weighted Sparse Coding
    Y_hat = zeros(par.ps2ch, par.maxrc, 'single');
    W_hat = zeros(par.ps2ch, par.maxrc, 'single');
    for i = 1:par.lenrc
        index = blk_arr(:, i);
        nlY = Y( : , index );
        DC = mean(nlY, 2);
        nDCnlY = bsxfun(@minus, nlY, DC);
        % update right weighting matrix W for weighted least square
        Wls = 1 ./ Sigma(index);
        % weighted least square and weighted sparse coding
        nDCnlYhat = WLSWSCN(nDCnlY, Wls, par);
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

