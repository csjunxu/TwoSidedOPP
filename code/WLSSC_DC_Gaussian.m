function  [im_out, par]    =   WLSSC_DC_Gaussian(par)
im_out    =   par.nim;
par.nSig0 = par.nSig;
% parameters for noisy image
[h,  w, ch]      =  size(im_out);
par.h = h;
par.w = w;
par.ch = ch;
par = SearchNeighborIndex( par );
NY = Image2PatchNew( par.nim, par);
par.TolN = size(NY, 2);
for ite  =  1 : par.outerIter
    % iterative regularization
    im_out = im_out + par.delta * (par.nim - im_out);
    % image to patches and estimate local noise variance
    Y = Image2PatchNew( im_out, par);
    % estimation of noise variance
    if mod(ite-1,par.innerIter)==0
        %         par.nlsp = par.nlsp - 10;
        % searching  non-local patches
        blk_arr = Block_Matching( Y, par);
    end
    if ite == 1
        par.nSig = par.nSig0;
    else
        dif = mean( mean( mean( (par.nim - im_out).^2 ) ) );
        par.nSig = sqrt( abs( par.nSig0^2 - dif ) )*par.lambda;
    end
    % Weighted Sparse Coding
    Y_hat = zeros(par.ps2ch, par.maxrc, 'single');
    W_hat = zeros(par.ps2ch, par.maxrc, 'single');
    for i = 1:par.lenrc
        index = blk_arr(:, i);
        Lindex = length(index);
        nlY = Y( : , index );
        DC = mean(nlY, 2);
        nDCnlY = bsxfun(@minus, nlY, DC);
        nDCnlYCh = [nDCnlY(1:par.ps2, :) nDCnlY(par.ps2+1:2*par.ps2, :) nDCnlY(2*par.ps2+1:3*par.ps2, :)];
        % Recovered Estimated Patches by weighted least square and weighted
        % sparse coding model
        nDCnlYhat = WLSSC_DC(nDCnlYCh, par);
        nDCnlYhat = [nDCnlYhat(:, 1:Lindex); nDCnlYhat(:, Lindex+1:2*Lindex); nDCnlYhat(:, 2*Lindex+1:3*Lindex)];
        % add DC components
        nlYhat = bsxfun(@plus, nDCnlYhat, DC);
        % aggregation
        Y_hat(:, index) = Y_hat(:, index) + nlYhat;
        W_hat(:, index) = W_hat(:, index) + ones(par.ps2ch, par.nlsp);
    end
    % Reconstruction
    im_out = PGs2Image(Y_hat, W_hat, par);
    % calculate the PSNR
    PSNR =   csnr( im_out, par.I, 0, 0 );
    SSIM      =  cal_ssim( im_out, par.I, 0, 0 );
    fprintf('Iter %d : PSNR = %2.4f, SSIM = %2.4f\n', ite, PSNR, SSIM);
    par.PSNR(ite, par.image) = PSNR;
    par.SSIM(ite, par.image) = SSIM;
end
return;

