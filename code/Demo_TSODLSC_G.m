clear;
Original_image_dir  =    'C:/Users/csjunxu/Desktop/JunXu/My_Paper/2015 ICCV_PGPD/testimages';
fpath = fullfile(Original_image_dir, '*.png');
im_dir  = dir(fpath);
im_num = length(im_dir);

nSig = 40;
par.ps = 7; % patch size
par.step = 6; % the step of two neighbor patches
par.win = 20;

par.outerIter = 6;
par.innerIter = 2;
par.Iter = 100;
par.epsilon = 0.01;

par.delta = 0;
for lambda = 5.5:0.5:8
    par.lambda = lambda;
    % record all the results in each iteration
    par.PSNR = zeros(par.outerIter, im_num, 'single');
    par.SSIM = zeros(par.outerIter, im_num, 'single');
    T512 = [];
    T256 = [];
    for i = 1:im_num
        par.nlsp = 70;  % number of non-local patches
        par.image = i;
        par.nSig = nSig/255;
        par.I =  single( imread(fullfile(Original_image_dir, im_dir(i).name)) )/255;
        S = regexp(im_dir(i).name, '\.', 'split');
        randn('seed',0);
        par.nim =   par.I + par.nSig*randn(size(par.I));
        %
        fprintf('%s :\n',im_dir(i).name);
        PSNR =   csnr( par.nim*255, par.I*255, 0, 0 );
        SSIM      =  cal_ssim( par.nim*255, par.I*255, 0, 0 );
        fprintf('The initial value of PSNR = %2.4f, SSIM = %2.4f \n', PSNR,SSIM);
        %
        time0 = clock;
        [im_out, par]  =  TSODLSC_denoising(par);
        fprintf('Total elapsed time = %f s\n', (etime(clock,time0)) );
        im_out(im_out>1)=1;
        im_out(im_out<0)=0;
        % calculate the PSNR
        par.PSNR(par.outerIter, par.image)  =   csnr( im_out*255, par.I*255, 0, 0 );
        par.SSIM(par.outerIter, par.image)      =  cal_ssim( im_out*255, par.I*255, 0, 0 );
        imname = sprintf(['TSODLSC_1AG_nSig' num2str(nSig) '_delta' num2str(par.delta) '_lsc' num2str(par.lambda) '_Iter' num2str(par.Iter) '_' im_dir(i).name]);
        imwrite(im_out,imname);
        fprintf('%s : PSNR = %2.4f, SSIM = %2.4f \n',im_dir(i).name, par.PSNR(par.outerIter, par.image),par.SSIM(par.outerIter, par.image)     );
    end
    mPSNR=mean(par.PSNR,2);
    [~, idx] = max(mPSNR);
    PSNR =par.PSNR(idx,:);
    SSIM = par.SSIM(idx,:);
    mSSIM=mean(SSIM,2);
    mT512 = mean(T512);
    sT512 = std(T512);
    mT256 = mean(T256);
    sT256 = std(T256);
    fprintf('The best PSNR result is at %d iteration. \n',idx);
    fprintf('The average PSNR = %2.4f, SSIM = %2.4f. \n', mPSNR(idx),mSSIM);
    name = sprintf(['TSODLSC_1AG_nSig' num2str(nSig) '_delta' num2str(par.delta) '_lsc' num2str(par.lambda) '_Iter' num2str(par.Iter) '.mat']);
    save(name,'nSig','PSNR','SSIM','mPSNR','mSSIM','mT512','sT512','mT256','sT256');
end