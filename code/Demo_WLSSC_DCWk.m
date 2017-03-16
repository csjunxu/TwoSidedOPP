clear;
Original_image_dir  =    'C:\Users\csjunxu\Desktop\JunXu\Datasets\kodak24\kodak_color\';
fpath = fullfile(Original_image_dir, '*.png');
im_dir  = dir(fpath);
im_num = length(im_dir);

nSig = 50;

% parameters
par.ps = 8;       % patch size
par.step = 7;    % the step of two neighbor patches
par.win = 20;   % size of window around the patch

par.outerIter = 8;
par.innerIter = 2;
par.epsilon = 0.01;
par.model = 2;
par.WWIter = 100;
par.delta = 0;

nlsp = 90;

par.method = 'WLSSC_DCWk_Gaussian';
for lambda = 0.5:0.1:1
    par.lambda = lambda;
    for lambdasc = [0.001]
        par.lambdasc = lambdasc;
        for lambdals = [0.5 1]
            par.lambdals = lambdals;
            % record all the results in each iteration
            par.PSNR = zeros(par.outerIter, im_num, 'single');
            par.SSIM = zeros(par.outerIter, im_num, 'single');
            for i = 1:im_num
                par.image = i;
                par.nlsp = nlsp;
                par.nSig = nSig;
                par.I =  double( imread(fullfile(Original_image_dir, im_dir(i).name)) );
                S = regexp(im_dir(i).name, '\.', 'split');
                [h, w, ch] = size(par.I);
                par.nim = zeros(size(par.I));
                for c = 1:ch
                    randn('seed',0);
                    par.nim(:, :, c) = par.I(:, :, c) + par.nSig * randn(size(par.I(:, :, c)));
                end
                %
                fprintf('%s :\n',im_dir(i).name);
                PSNR =   csnr( par.nim, par.I, 0, 0 );
                SSIM      =  cal_ssim( par.nim, par.I, 0, 0 );
                fprintf('The initial value of PSNR = %2.4f, SSIM = %2.4f \n', PSNR,SSIM);
                %
                time0 = clock;
                im_out = WLSSC_DCWk_Gaussian( par ); % WNNM denoisng function
                fprintf('Total elapsed time = %f s\n', (etime(clock,time0)) );
                im_out(im_out>255)=255;
                im_out(im_out<0)=0;
                % calculate the PSNR
                par.PSNR(par.outerIter, par.image)  =   csnr( im_out, par.I, 0, 0 );
                par.SSIM(par.outerIter, par.image)      =  cal_ssim( im_out, par.I, 0, 0 );
                %             imname = sprintf('nSig%d_clsnum%d_delta%2.2f_lambda%2.2f_%s', nSig, cls_num, delta, lambda, im_dir(i).name);
                %             imwrite(im_out,imname);
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
            name = sprintf([par.method '_' num2str(im_num) '_nSig' num2str(nSig) '_lambda' num2str(lambda) '_lambdasc' num2str(lambdasc) '_lambdals' num2str(lambdals) '.mat']);
            save(name,'nSig','PSNR','SSIM','mPSNR','mSSIM');
        end
    end
end