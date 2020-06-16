close all;clear all;clc;
pf=0.005;  %已定的虚警概率
snr_db=-50:1:20;%信噪比
snr=10.^(snr_db/10);%db转化十进制
unc_db=[0,0.01,0.1,0.5,1,3];%不确定度
unc=10.^(unc_db/10);
N=[100,1000,10000];
n=N(1);
[Pd_k,PD_k]=knn_unc(unc,snr_db,1000);
[Pd_e,PD_e]=ED_unc(unc,snr_db,1000);

ED_1=PD_e(1,:);ED_2=PD_e(2,:);ED_3=PD_e(3,:);
ED_4=PD_e(4,:);ED_5=PD_e(5,:);ED_6=PD_e(6,:);
E_N=[ED_1;ED_2;ED_3;ED_4;ED_5;ED_6];

KNN_1=PD_k(1,:);KNN_2=PD_k(2,:);KNN_3=PD_k(3,:);
KNN_4=PD_k(4,:);KNN_5=PD_k(5,:);KNN_6=PD_k(6,:);
K_N=[KNN_1;KNN_2;KNN_3;KNN_4;KNN_5;KNN_6];

pk=plot(snr_db,K_N,'-o',snr_db,E_N,'-*');
grid minor;
legend(pk(1:6),'KNN x=0dB','KNN x=0.01dB','KNN x=0.1dB','KNN x=0.5dB','KNN x=1dB','KNN x=3dB');
ah=axes('position',get(gca,'position'),...
          'visible','off');
legend(ah,pk(7:12),'ED x=0dB','ED x=0.01dB','ED x=0.1dB','ED x=0.5dB','ED x=1dB','ED x=3dB');
xlabel('SNR/dB');ylabel('Probability of detection');

% for i=1:size(PD_k,1)
%     plot(snr_db,PD_e(i,:),'-o')
%     hold on;
%     plot(snr_db,PD_k(i,:),'-*')
% end
% grid minor;
% xlabel('SNR/dB');ylabel('Probability of detection');

































% x=0.01;r=10.^(x/10);
% Nn=2000;snrr_db=-10;sr=10.^(snrr_db/10);
% pf=0:0.01:1;
% pd=qfunc(qfuncinv(pf)-sqrt(0.5*Nn)*sr);
% plot(pf,1-pd,'k--','linewidth',2.5);axis([0 0.5 0 0.5]);
% xlabel('Probability of False Alarm ');ylabel('Probability of Missed-Detection');