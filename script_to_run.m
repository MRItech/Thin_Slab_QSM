% script to compare thin slab reconstruction using:
%              1) Standard pipeline with MEDI inversion
%              2) Proposed method with low-res "tissue field" regularization
%              3) Proposed method with low-res "susceptibility" regulariz.
%
% By Nashwan Naji, nashwana@ualberta.ca

function script_to_run
 clear
 % add code files to matlab search path
 addpath(genpath(pwd))
%% settings
 recon_method ={'Standard','Proposed with R1(x)','Proposed with R2(x)'}; no_methods = length(recon_method);
% recon_method ={'Proposed with R1(x)','Proposed with R2(x)'};

fov_pc =  [100  50  20 10  5  2 1 0.5]; % slab thickness as % of full width
fov_slc = [386  180 90 44  22 8 4  2];  % slab thickness as number of slices
no_slabs = length(fov_slc);

slice_1 = 177; % GP,PT,CD,TH, Hemorrhage and Calcification
slice_2 = 140; % SN and RN
slice_location = 'PT';  % 'PT': centered on Putamen, or 'RN': centered on Red nucleus 
select_low_res_echoTimes =1:6;      % select TEs to be used from low-res
use_GPU_for_dipole_inversion = 1;   % use GPU if available
 %% data path
data_path = 'data/';
hi_res_path = [data_path,'hi_res/'];
low_res_path = [data_path,'low_res_x4x4x4/'];

%% constants
  
B0 = 3;                            % field strength [T] used in simulation
z_prjs = [ 0 0 1];                 % main field direction
TE = [4.6 11.7 18.8 25.9 33 40.1]; % echo-times [ms] used in simulation 

CF = B0*42.5750; % imaging freqency [MHz], used for scaling

if strcmp(slice_location,'RN')
    afx = '_RN';
    vis_slice = slice_2;
else
    afx ='_PT';
    vis_slice = slice_1;
end
legend_txt {1}= 'Ref. (Standard at full width)'; 
%% load data

% hi-res

nii = load_nii([hi_res_path,'mag_sqrt.nii']); mag_sqrt= single(nii.img);
vox = nii.hdr.dime.pixdim(2:4); 
nii = load_nii([hi_res_path,'tfs.nii']); tfs= double(nii.img);
nii = load_nii([hi_res_path,'lfs_mask.nii']); lfs_mask= single(nii.img);

nii = load_nii([data_path,'segmentation_labels.nii']); seg= single(nii.img);
%  one slice segmentation for 2D measurements
seg_2d = seg; seg_2d(:,:,1:slice_1(1)-1)=0; seg_2d(:,:,slice_1(end)+1:end)=0;
seg_2d2 = seg; seg_2d2(:,:,1:slice_2(1)-1)=0; seg_2d2(:,:,slice_2(end)+1:end)=0;
nii = load_nii([hi_res_path,'mask.nii']); mask= single(nii.img);


% low-res
nii = load_nii([low_res_path,'\mag.nii']); mag_low= single(nii.img(:,:,:,select_low_res_echoTimes));
nii = load_nii([low_res_path,'\phase.nii']); phase_low= single(nii.img(:,:,:,select_low_res_echoTimes));
vox_low = nii.hdr.dime.pixdim(2:4); 

no_echo = size(mag_low,4); % number of echoes used from the low-res data

downsamplin_factor = vox_low./vox;  % resolution ratio between the low- and hi-res data

%% ================================================================================= %%
%                          Preparation: Processing Low-res data
% ===================================================================================
%% up-sample the low-res data 
cmpx = mag_low.*exp(1j.*phase_low); clear mag_low phase_low

for i = 1: no_echo
    cmx_intrp(:,:,:,i) = m1_upsampl_n(cmpx(:,:,:,i),downsamplin_factor);
end

mag_low_intrp = abs(cmx_intrp);
phase_low_intrp = angle(cmx_intrp); clear cmx_intrp cmpx

%% unwrap the up-sampled phase of low-res data using ROMEO
% ROMEO phase unwrapping can be downloaded from: https://github.com/korbinian90/ROMEO

parameters.output_dir = fullfile(['romeo_tmp']); 
parameters.TE = TE(select_low_res_echoTimes); % required for multi-echo
parameters.mag = mag_low_intrp;
parameters.mask = mask;  
parameters.calculate_B0 = true; % optianal B0 calculation for multi-echo
parameters.phase_offset_correction = 'off'; % options are: 'off' | 'on' | 'bipolar'
parameters.voxel_size = vox; 
parameters.additional_flags = '--verbose'; % settings are pasted directly to ROMEO cmd (see https://github.com/korbinian90/ROMEO for options)

mkdir(parameters.output_dir);

[unphase_low_intrp, ~] = ROMEO(phase_low_intrp, parameters); clear phase_low_intrp


%% masking less relaible phase

    echo_mask = mag_low_intrp;
    A=mag_low_intrp(:,:,:,1);
    Q = quantile(A(:),0.7); A(A<Q) = nan;
    max_mag = mean(A(:),'omitnan');
    echo_mask(echo_mask<=max_mag/2) = 0;
    echo_mask(echo_mask~=0) = 1;
    
    echo_mask = imopen(echo_mask,strel('cube',11) );
    echo_mask(:,:,:,1)= parameters.mask; echo_mask(:,:,:,2)= parameters.mask;
     

%% compute total field shift for the low-res data

TE4D = repmat(permute(TE,[5 4 3 2 1]),[size(mag_low_intrp,1:3) 1]);
tfs_low = (1000/2/pi)* sum(unphase_low_intrp.*mag_low_intrp.*mag_low_intrp.*echo_mask.*TE4D,4 )./sum(mag_low_intrp.*mag_low_intrp.*echo_mask.*TE4D.^2,4);  
tfs_low(isnan(tfs_low))=0;
tfs_low(isinf(tfs_low))=0;

% V-SHARP from STI Suite V3.0, https://people.eecs.berkeley.edu/~chunlei.liu/software.html
[lfs_low,lfs_mask_low]=V_SHARP(tfs_low.*echo_mask(:,:,:,1),echo_mask(:,:,:,1),'voxelsize',vox,'smvsize',20);
bkg_low = (tfs_low - lfs_low).*lfs_mask_low;

% remove voxels outside brain, to reduce memory demand
bkg_low = bkg_low(51:456, 46:566,21:406);
lfs_low = lfs_low(51:456, 46:566,21:406);
lfs_mask_low = lfs_mask_low(51:456, 46:566,21:406);
mag_low_intrp = mag_low_intrp(51:456, 46:566,21:406,:);

N= size(mag_low_intrp);
mag_sqrt_low = sqrt(sum(abs(mag_low_intrp).^2,4)); clear mag_low_intrp unphase_low_intrp

%% recon suseptibiltiy for low-res data

sus_low_file = ['output/sus_low.nii'];
if~exist(sus_low_file,'file')

            options.tfs = lfs_low/CF;
            options.mag =  mag_sqrt_low;
            options.mask = lfs_mask_low;
            options.N_std = 1./options.mag;
            options.voxel_size = vox;
            options.B0_dir = z_prjs;
            options.lambda = 50000; 
            options.useGPU =use_GPU_for_dipole_inversion;
            sus_low =  MEDI_linear_gpu_sub(options); % run
            sus_low = sus_low.x.*options.mask; clear options
            
nii = make_nii(single(sus_low),vox);save_nii(nii,sus_low_file);
else
    nii = load_nii(sus_low_file); sus_low = single(nii.img);
end
%% ================================================================================= %%
%                          Start Reconstruction
% ===================================================================================

% to store measurements
mean_sus_2d_PT = zeros(no_slabs,4,10);
error_sus2d_PT = zeros(no_slabs,4,10);
mean_sus_2d_RN = zeros(no_slabs,4,10);
error_sus2d_RN = zeros(no_slabs,4,10);

% to store sample images for visualization
to_visualize = zeros([N(1:2) no_slabs no_methods], 'single');

for method_i = 1:no_methods

    current_method = recon_method{method_i};
    disp ('% ================================================================================= %')
    disp (['        ',num2str(method_i), ' ) applying ', current_method, '      ....'])

    legend_txt{method_i+1} = current_method;

    %% Loop for different slab thicknesses

    for sel_FOV =1: no_slabs
    
        disp(['FOV ',num2str(fov_pc(sel_FOV)),'% .............'])

       if fov_pc(sel_FOV) == 100 % full width
            cnt =  N(3)/2;
        else
            if strcmp(slice_location,'RN')
                cnt = slice_2;
            else
                cnt = slice_1;
            end
       end

       % slab mask
        fov_maks = zeros(size(lfs_mask),'single');
        fov_maks(:,:, cnt - fov_slc(sel_FOV)/2 +1: cnt +fov_slc(sel_FOV)/2) = 1;


        rtfs = tfs.*fov_maks;  % reduce the width of the total field
        rmask_lfs = lfs_mask.*fov_maks; % reduce the width of the mask
        rmask = mask(51:456, 46:566,21:406).*fov_maks;

    %% background field removal
        clear rlfs
    switch current_method
        case 'Standard'   % remove using V-SHARP technique
            if fov_slc(sel_FOV)*vox(1)/2 >= 20
                rad = 20;
            else
                rad = ceil(fov_slc(sel_FOV)*vox(1)/2);
            end

            try
                [rlfs,~]=V_SHARP(rtfs.*rmask,rmask,'voxelsize',vox,'smvsize',rad);
            catch
                rlfs = nan(N(1:3));
                disp('Warning! V-SHARP failed due to the too thin slab.')
            end

        prnt = 'Standard';

        case {'Proposed with R1(x)','Proposed with R2(x)'} % romve using proposed method
                
            % obtain tissue field by subtracting bkg field estimated from
            % the low-res data
            disp('Remove background field by subtraction.')

        rlfs = single((rtfs - bkg_low).*fov_maks.*rmask_lfs); 

        if strcmp(current_method,'Proposed with R1(x)')
            prnt = 'Proposed_lfs';
        else
            prnt = 'Proposed_sus';
        end


        otherwise
            prnt ='error';
    end


    
 save_path = ['output/','FOV_',num2str(fov_slc(sel_FOV)),'slices_',prnt,'/'];
mkdir(save_path)
nii = make_nii(single(rlfs),vox);save_nii(nii,[save_path,'lfs_red',afx,'.nii']);


    %% dipole inversion using MEDI

            options.tfs = rlfs/CF;
            options.mag =  mag_sqrt.*rmask_lfs;
            options.mask = rmask_lfs;
            options.N_std = 1./options.mag;
            options.voxel_size = vox;
            options.B0_dir = z_prjs;
            options.lambda = 50000; 
            options.useGPU =use_GPU_for_dipole_inversion;

            switch current_method
                case 'Standard'  % dont use low-res information
                    % jump to running the dipole inversion
                
                case 'Proposed with R1(x)'   
                    options.mask_fovi = fov_maks; % slab mask
                    options.lfs_low = lfs_low/CF; % low-res tissue field
                    options.mag_low = mag_sqrt_low; % mag
                    options.downsampl_factor = downsamplin_factor;
                    options.lambda2 = options.lambda/2; 

                case 'Proposed with R2(x)'
                    options.mask_fovi = fov_maks; % slab mask
                    options.sus_low = sus_low;  % low-res susceptibiltiy
                    options.mag_low = mag_sqrt_low; % mag
                    options.downsampl_factor = downsamplin_factor;
                    options.lambda2 = options.lambda/2; 

                otherwise
                    disp('Unkown option! ')
            end
            
            % run dipole inversion 
            sus =  MEDI_linear_gpu_sub(options); % run

            sus = sus.x.*rmask_lfs; clear options
            nii = make_nii(sus,vox);save_nii(nii,[save_path,'sus_medi',afx,'.nii']);

            % store one slice for visualization
            to_visualize(:,:,sel_FOV,method_i) = single(sus(:,:,vis_slice));
%% ================================================================================= %%
%                         Measurements
% ===================================================================================

if strcmp(current_method,'Standard') && fov_pc(sel_FOV) == 100 % Standard & full width
    full_sus = sus;
end

        cnt2 = 1;

    for k = [1 3 5 7 11 13 15 17 19 20]


        str_mask_2d = zeros(size(seg));
        str_mask_2d2 = zeros(size(seg));


        str_mask_2d(seg_2d==k)=1;
        str_mask_2d2(seg_2d2==k)=1;

        if k<19  % 19: hemorrhage, 20: calcification; i.e. no left/right 

            str_mask_2d(seg_2d==k+1)=1;
            str_mask_2d2(seg_2d2==k+1)=1;
        end
       
        switch current_method
        
        
            case'Standard'
                % ========= Ref, full-width, using standard method
                roi_mask_2d = str_mask_2d.*fov_maks;
                roi_sus= full_sus.*roi_mask_2d;
                mean_sus_2d_PT(sel_FOV,1,cnt2) = sum(roi_sus(:).*roi_mask_2d(:))/sum(roi_mask_2d(:));

                roi_mask_2d = str_mask_2d2.*fov_maks;
                roi_sus= full_sus.*roi_mask_2d;
                mean_sus_2d_RN(sel_FOV,1,cnt2) = sum(roi_sus(:).*roi_mask_2d(:))/sum(roi_mask_2d(:));
        
        
                % ========= Standard method at different slab widthes
                if ~strcmp(slice_location, 'RN')
                roi_mask_2d = str_mask_2d.*fov_maks;
                roi_sus= sus.*roi_mask_2d;
                mean_sus_2d_PT(sel_FOV,2,cnt2) = sum(roi_sus(:).*roi_mask_2d(:))/sum(roi_mask_2d(:));
                error_sus2d_PT(sel_FOV,2,cnt2) = abs(mean_sus_2d_PT(sel_FOV,1,cnt2) - mean_sus_2d_PT(sel_FOV,2,cnt2));
        
                else
                roi_mask_2d = str_mask_2d2.*fov_maks;
                roi_sus= sus.*roi_mask_2d;
                mean_sus_2d_RN(sel_FOV,2,cnt2) = sum(roi_sus(:).*roi_mask_2d(:))/sum(roi_mask_2d(:));
                error_sus2d_RN(sel_FOV,2,cnt2) = abs(mean_sus_2d_RN(sel_FOV,1,cnt2) - mean_sus_2d_RN(sel_FOV,2,cnt2));
                end

            case 'Proposed with R1(x)'
                % ======== Proposed method with tissue field [i.e., R1(x)] 
                if ~strcmp(slice_location, 'RN')
                roi_mask_2d = str_mask_2d.*fov_maks;
                roi_sus= sus.*roi_mask_2d;
                mean_sus_2d_PT(sel_FOV,3,cnt2) = sum(roi_sus(:).*roi_mask_2d(:))/sum(roi_mask_2d(:)) ;%+ low_fov_sus;
                error_sus2d_PT(sel_FOV,3,cnt2) = abs(mean_sus_2d_PT(sel_FOV,1,cnt2) - mean_sus_2d_PT(sel_FOV,3,cnt2));
        
                else
                roi_mask_2d = str_mask_2d2.*fov_maks;
                roi_sus= sus.*roi_mask_2d;
                mean_sus_2d_RN(sel_FOV,3,cnt2) = sum(roi_sus(:).*roi_mask_2d(:))/sum(roi_mask_2d(:)) ;%+ low_fov_sus;
                error_sus2d_RN(sel_FOV,3,cnt2) = abs(mean_sus_2d_RN(sel_FOV,1,cnt2) - mean_sus_2d_RN(sel_FOV,3,cnt2));
                end

            case 'Proposed with R2(x)'
                % ======== Proposed method with susceptibiltiy [i.e., R2(x)] 
                if ~strcmp(slice_location, 'RN')
                roi_mask_2d = str_mask_2d.*fov_maks;
                roi_sus= sus.*roi_mask_2d;
                mean_sus_2d_PT(sel_FOV,4,cnt2) = sum(roi_sus(:).*roi_mask_2d(:))/sum(roi_mask_2d(:)) ;%+ low_fov_sus;
                error_sus2d_PT(sel_FOV,4,cnt2) = abs(mean_sus_2d_PT(sel_FOV,1,cnt2) - mean_sus_2d_PT(sel_FOV,4,cnt2));

                else
                roi_mask_2d = str_mask_2d2.*fov_maks;
                roi_sus= sus.*roi_mask_2d;
                mean_sus_2d_RN(sel_FOV,4,cnt2) = sum(roi_sus(:).*roi_mask_2d(:))/sum(roi_mask_2d(:)) ;%+ low_fov_sus;
                error_sus2d_RN(sel_FOV,4,cnt2) = abs(mean_sus_2d_RN(sel_FOV,1,cnt2) - mean_sus_2d_RN(sel_FOV,4,cnt2));
                end
        end
        cnt2 = cnt2 + 1;
    end

end

end
%% ================================================================================= %%
%                          Plot Results
% ===================================================================================

error_sus2d_pc_PT = error_sus2d_PT./abs(mean_sus_2d_PT)*100;  % error as % in slice 1
error_sus2d_pc_RN = error_sus2d_RN./abs(mean_sus_2d_RN)*100; % error as % in slice 2

x_ax = repmat(fov_slc.',1,4); % x-axis for the plots


figure(2019)
switch slice_location
    case 'PT'
x1=subplot(3,3,1); gp=plot(x_ax,mean_sus_2d_PT(:,:,8)-mean_sus_2d_PT(:,:,1),'-o','LineWidth',2,'MarkerSize',4); title('Globus pallidus'), grid on, ylim([-.2 .35])
x2=subplot(3,3,2); pt=plot(x_ax,mean_sus_2d_PT(:,:,7)-mean_sus_2d_PT(:,:,1),'-o','LineWidth',2,'MarkerSize',4); title('Putamen'), grid on, ylim([-.15 .3])
x3=subplot(3,3,3); cd=plot(x_ax,mean_sus_2d_PT(:,:,6)-mean_sus_2d_PT(:,:,1),'-o','LineWidth',2,'MarkerSize',4); title('Caudate'), grid on, ylim([-.15 .2])
x4=subplot(3,3,4); th=plot(x_ax,mean_sus_2d_PT(:,:,5)-mean_sus_2d_PT(:,:,1),'-o','LineWidth',2,'MarkerSize',4); title('Thalamus'), grid on, ylim([-.2 .15])
x5=subplot(3,3,5); hm=plot(x_ax,mean_sus_2d_PT(:,:,10)-mean_sus_2d_PT(:,:,1),'-o','LineWidth',2,'MarkerSize',4); title('Hemorrhage'), grid on, ylim([-.2 1])
x6=subplot(3,3,6); cl=plot(x_ax,mean_sus_2d_PT(:,:,9)-mean_sus_2d_PT(:,:,1),'-o','LineWidth',2,'MarkerSize',4); title('Calcification'), grid on, ylim([-.6 .05])
    case 'RN'
x7=subplot(3,3,7); sn=plot(x_ax,mean_sus_2d_RN(:,:,3)-mean_sus_2d_RN(:,:,1),'-o','LineWidth',2,'MarkerSize',4); title('Substantia nigra '), grid on, ylim([-.065 .266])
x8=subplot(3,3,8); rn=plot(x_ax,mean_sus_2d_RN(:,:,2)-mean_sus_2d_RN(:,:,1),'-o','LineWidth',2,'MarkerSize',4); title('Red nucleus'), grid on, ylim([-.15 .35])
end

legend(legend_txt)
% styling
switch slice_location
    case 'PT'
gp(1).LineStyle ='--';
gp(1).Marker = 'none';
gp(1).Color = [0.5020    0.5020    0.5020];
gp(2).LineStyle =':';
gp(2).Marker = '*';
gp(2).Color = '#4DBEEE';
gp(2).MarkerSize = 4;

gp(3).LineStyle ='-';
gp(3).Marker = 'o';
gp(3).Color = '#FF0000';
gp(3).MarkerSize = 4;
gp(4).LineStyle ='-';
gp(4).Marker = 'd';
gp(4).Color = '#7E2F8E';
gp(4).MarkerSize = 3;

pt(1).LineStyle ='--';
pt(1).Marker = 'none';
pt(1).Color = [0.5020    0.5020    0.5020];
pt(2).LineStyle =':';
pt(2).Marker = '*';
pt(2).Color = '#4DBEEE';
pt(2).MarkerSize = 4;

pt(3).LineStyle ='-';
pt(3).Marker = 'o';
pt(3).Color = '#FF0000';
pt(3).MarkerSize = 4;
pt(4).LineStyle ='-';
pt(4).Marker = 'd';
pt(4).Color = '#7E2F8E';
pt(4).MarkerSize = 3;

cd(1).LineStyle ='--';
cd(1).Marker = 'none';
cd(1).Color = [0.5020    0.5020    0.5020];
cd(2).LineStyle =':';
cd(2).Marker = '*';
cd(2).Color = '#4DBEEE';
cd(2).MarkerSize = 4;

cd(3).LineStyle ='-';
cd(3).Marker = 'o';
cd(3).Color = '#FF0000';
cd(3).MarkerSize = 4;
cd(4).LineStyle ='-';
cd(4).Marker = 'd';
cd(4).Color = '#7E2F8E';
cd(4).MarkerSize = 3;

th(1).LineStyle ='--';
th(1).Marker = 'none';
th(1).Color = [0.5020    0.5020    0.5020];
th(2).LineStyle =':';
th(2).Marker = '*';
th(2).Color = '#4DBEEE';
th(2).MarkerSize = 4;

th(3).LineStyle ='-';
th(3).Marker = 'o';
th(3).Color = '#FF0000';
th(3).MarkerSize = 4;
th(4).LineStyle ='-';
th(4).Marker = 'd';
th(4).Color = '#7E2F8E';
th(4).MarkerSize = 3;

hm(1).LineStyle ='--';
hm(1).Marker = 'none';
hm(1).Color = [0.5020    0.5020    0.5020];
hm(2).LineStyle =':';
hm(2).Marker = '*';
hm(2).Color = '#4DBEEE';
hm(2).MarkerSize = 4;

hm(3).LineStyle ='-';
hm(3).Marker = 'o';
hm(3).Color = '#FF0000';
hm(3).MarkerSize = 4;
hm(4).LineStyle ='-';
hm(4).Marker = 'd';
hm(4).Color = '#7E2F8E';
hm(4).MarkerSize = 3;

cl(1).LineStyle ='--';
cl(1).Marker = 'none';
cl(1).Color = [0.5020    0.5020    0.5020];
cl(2).LineStyle =':';
cl(2).Marker = '*';
cl(2).Color = '#4DBEEE';
cl(2).MarkerSize = 4;

cl(3).LineStyle ='-';
cl(3).Marker = 'o';
cl(3).Color = '#FF0000';
cl(3).MarkerSize = 4;
cl(4).LineStyle ='-';
cl(4).Marker = 'd';
cl(4).Color = '#7E2F8E';
cl(4).MarkerSize = 3;

x1.FontName ='Cambria';
x1.FontSize =12;
x2.FontName ='Cambria';
x2.FontSize =12;
x3.FontName ='Cambria';
x3.FontSize =12;
x4.FontName ='Cambria';
x4.FontSize =12;
x5.FontName ='Cambria';
x5.FontSize =12;
x6.FontName ='Cambria';
x6.FontSize =12;
    case 'RN'
sn(1).LineStyle ='--';
sn(1).Marker = 'none';
sn(1).Color = [0.5020    0.5020    0.5020];
sn(2).LineStyle =':';
sn(2).Marker = '*';
sn(2).Color = '#4DBEEE';
sn(2).MarkerSize = 4;

sn(3).LineStyle ='-';
sn(3).Marker = 'o';
sn(3).Color = '#FF0000';
sn(3).MarkerSize = 4;
sn(4).LineStyle ='-';
sn(4).Marker = 'd';
sn(4).Color = '#7E2F8E';
sn(4).MarkerSize = 3;

rn(1).LineStyle ='--';
rn(1).Marker = 'none';
rn(1).Color = [0.5020    0.5020    0.5020];
rn(2).LineStyle =':';
rn(2).Marker = '*';
rn(2).Color = '#4DBEEE';
rn(2).MarkerSize = 4;

rn(3).LineStyle ='-';
rn(3).Marker = 'o';
rn(3).Color = '#FF0000';
rn(3).MarkerSize = 4;
rn(4).LineStyle ='-';
rn(4).Marker = 'd';
rn(4).Color = '#7E2F8E';
rn(4).MarkerSize = 3;

x7.FontName ='Cambria';
x7.FontSize =12;
x8.FontName ='Cambria';
x8.FontSize =12;
end

%% ================================================================================= %%
%                          Show recon. QSM at different slab widths 
% ===================================================================================
if no_slabs > 4
    sel_slab_for_show = [fov_slc(end:-2:1) fov_slc(1)]; slb_indx = [no_slabs:-2:1 1];
    sel_slab_for_show = sel_slab_for_show(end:-1:1); slb_indx= slb_indx(end:-1:1);
else
    sel_slab_for_show = fov_slc; slb_indx = 1:no_slabs;
end
no_shows = length(sel_slab_for_show);

figure(2020)
t = tiledlayout(no_methods,no_shows,'TileSpacing','Compact','Padding','Compact');
for method_i = 1: no_methods
    for slb_i = 1: no_shows
        
    ax=nexttile; imshow(to_visualize(:,:,slb_indx(slb_i),method_i).', [-.10 .20]), title([num2str(sel_slab_for_show(slb_i)), '-slice slab (',num2str(sel_slab_for_show(slb_i)*vox(3)),'mm)'])
        
        if slb_i ==1
            ylabel(recon_method{method_i},'FontSize',12,'FontWeight','bold','Color','b')
        end
    end
end


end
%% ================================================================================= %%
%                          Sub Functions
% ===================================================================================

function ima_hires = m1_upsampl_n(ima,upsampl_vec)
% Upsample image resolution
% Created by Julio Acosta-Cabronero, QSMBox: https://gitlab.com/acostaj/QSMbox


matrix_size             = size(ima);
dim1                    = matrix_size(1);
dim2                    = matrix_size(2);
dim3                    = matrix_size(3);
kfdoff                  = 30; % # of voxels where the filter will drop off from 1 to 0.
[kf1,kf2,kf3]           = ndgrid(m1_cosfilt(dim1,kfdoff),m1_cosfilt(dim2,kfdoff),m1_cosfilt(dim3,kfdoff));
kf                      = min(min(kf1,kf2),kf3); clear kf1 kf2 kf3

new_matrix_size         = round(matrix_size.*upsampl_vec);
pad                     = (new_matrix_size-matrix_size)/2;
rem_pad                 = rem(new_matrix_size-matrix_size,2);
rem_pad(rem_pad==.5)    = 1;
pad                     = floor(pad); rem_pad=0;

ima = ifftshift(ima);                         clear S
B = fftn(ima);                                clear A
K = fftshift(B);                            clear B


        

        K_real = kf.*real(K);
        K_real_pad = padarray(K_real,pad,'both');   clear K_real
        if sum(rem_pad(:))>0
            K_real_pad = padarray(K_real_pad,rem_pad,'pre');
        end
        K_imag = kf.*imag(K);                       clear K
        K_imag_pad = padarray(K_imag,pad,'both');   clear K_imag kf
        if sum(rem_pad(:))>0
            K_imag_pad = padarray(K_imag_pad,rem_pad,'pre');
        end
        K_pad = complex(K_real_pad,K_imag_pad);     clear K_*_pad
        A = ifftshift(K_pad);                       clear K_pad
        B = ifftn(A);                               clear A
        ima_hires = fftshift(B);                    clear B


end

  function kf = m1_cosfilt(L,doff)
% 1-D cosine filter

kf                      = ones(L+2,1);
kf(end-doff-1:end,1)    = cos(linspace(0,pi/2,doff+2));
kf(1:doff+2)            = cos(linspace(pi/2,0,doff+2));
kf                      = kf(2:end-1).^2;
  end
