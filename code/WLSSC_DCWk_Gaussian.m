function  [im_out, par]    =   WLSSC_DCWk_Gaussian(par)
im_out    =   par.nim;
par.nSig0 = par.nSig;
% parameters for noisy image
[h,  w, ch]      =  size(im_out);
par.h = h;
par.w = w;
par.ch = ch;
par = SearchNeighborIndex( par );
NY = Image2PatchNew( par.nim, par);
NYCh = [NY(1:par.ps2, :) NY(par.ps2+1:2*par.ps2, :) NY(2*par.ps2+1:3*par.ps2, :)];
par.TolN = size(NY, 2);
for ite  =  1 : par.outerIter
    % iterative regularization
    im_out = im_out + par.delta * (par.nim - im_out);
    % image to patches and estimate local noise variance
    Y = Image2PatchNew( im_out, par);
    YCh = [Y(1:par.ps2, :) Y(par.ps2+1:2*par.ps2, :) Y(2*par.ps2+1:3*par.ps2, :)];
    Sigma = par.lambda * sqrt(abs(repmat(par.nSig^2, 1, size(YCh, 2)) - mean((NYCh - YCh).^2))); %Estimated Local Noise Level
    % estimation of noise variance
    if mod(ite-1,par.innerIter)==0
        %         par.nlsp = par.nlsp - 10;
        % searching  non-local patches
        blk_arr = Block_Matching( Y, par);
        if ite == 1
            Wls = exp( - par.lambdals * sum(Sigma .^2, 1) );
        end
    end
    % Weighted Sparse Coding
    Y_hat = zeros(par.ps2ch, par.maxrc, 'single');
    W_hat = zeros(par.ps2ch, par.maxrc, 'single');
    for i = 1:par.lenrc
        index = blk_arr(:, i);
        Lindex = length(index);
        indexCh = [index; index+par.TolN; index+2*par.TolN];
        nlY = Y( : , index );
        DC = mean(nlY, 2);
        nDCnlY = bsxfun(@minus, nlY, DC);
        nDCnlYCh = [nDCnlY(1:par.ps2, :) nDCnlY(par.ps2+1:2*par.ps2, :) nDCnlY(2*par.ps2+1:3*par.ps2, :)];
        % Recovered Estimated Patches by weighted least square and weighted
        % sparse coding model
        nDCnlYhatCh = WLSSC_DCWk(nDCnlYCh, Sigma(indexCh), Wls(indexCh), par);
        % update weight for least square
        Wls(indexCh) = exp( - par.lambdals * sum((nDCnlYCh - nDCnlYhatCh) .^2, 1) );
        nDCnlYhat = [nDCnlYhatCh(:, 1:Lindex); nDCnlYhatCh(:, Lindex+1:2*Lindex); nDCnlYhatCh(:, 2*Lindex+1:3*Lindex)];
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

