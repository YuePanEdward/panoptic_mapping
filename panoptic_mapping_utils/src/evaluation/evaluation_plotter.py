#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Experiment results
# MAE_voxblox = np.array([0.330298, 0.257361 ,0.207521, 0.177987, 0.137868]) * 100.0
# Time_voxblox = np.array([0.06, 0.09, 0.20, 0.33, 0.58])

# MAE_cblox = np.array([0.329439, 0.271832, 0.250282, 0.194661, 0.150266, 0.128217]) * 100.0
# Time_cblox = np.array([0.01, 0.02 ,0.04, 0.09, 0.21, 0.39])

# MAE_panmap = np.array([0.140018, 0.117214, 0.071484, 0.061302, 0.0428536, 0.029744]) * 100.0 
# Time_panmap = np.array([0.08, 0.1, 0.13, 0.15, 0.22, 0.30])


# # Plot the results

# plt.rcParams.update({'font.size': 24})

# line_voxblox = plt.plot(Time_voxblox,  MAE_voxblox, '-s', linewidth=3, markersize=12)
# line_cblox = plt.plot(Time_cblox, MAE_cblox, '-o', linewidth=3, markersize=12)
# line_panmap = plt.plot(Time_panmap,  MAE_panmap, '-^', linewidth=3, markersize=12)
# line_realtime = plt.plot([0.1, 0.1], [0, 35], '--k', linewidth=2)

# plt.ylim((0, 35))
# plt.xlim((0, 0.6))
# plt.xlabel('Time per frame (s)')
# plt.ylabel('Average reconstruction error (cm)')
# plt.legend(['Voxblox [1]', 'Cblox [2]', 'Multi-resolution panmap (Ours)'])

# plt.show()


# Flat dataset experiment results
flat_tsdf_ours = np.array([0.010583, 0.024113, 0.037137, 0.051442, 0.072433]) * 100.0
flat_tsdf_voxblox = np.array([0.014940, 0.031971, 0.044054, 0.062959, 0.088687]) * 100.0

flat_mesh_ours = np.array([0.007675, 0.012193, 0.019522, 0.024548, 0.040639]) * 100.0
flat_mesh_voxblox = np.array([0.008339, 0.013382, 0.019278, 0.023516, 0.040656]) * 100.0

flat_coverage_ours = [76.91, 81.97, 84.02, 87.34, 90.46]
flat_coverage_voxblox = [75.58, 80.41, 82.85, 85.31, 88.23]

flat_esdf_occ_ours = np.array([0.000264, 0.001312, 0.002830, 0.005415, 0.006537]) * 100.0
flat_esdf_occ_ours_2 = np.array([0.0238, 0.044681, 0.059845, 0.071851, 0.100507]) * 100.0
flat_esdf_occ_voxblox = np.array([0.028618, 0.049641, 0.060068, 0.073047, 0.122605]) * 100.0
flat_esdf_occ_fiesta = np.array([0.000784, 0.002006, 0.002205, 0.005456, 0.005863]) * 100.0
flat_esdf_occ_edt = np.array([0.003646, 0.007013, 0.009817, 0.011729, 0.008791]) * 100.0

flat_esdf_gt_ours = np.array([0.013758, 0.026618, 0.039706, 0.051136, 0.065019]) * 100.0
flat_esdf_gt_ours_2 = np.array([0.022674, 0.042732, 0.055756, 0.070484, 0.105568]) * 100.0
flat_esdf_gt_voxblox = np.array([0.035935, 0.059597, 0.064879, 0.083333, 0.115038]) * 100.0
flat_esdf_gt_fiesta = np.array([0.021043, 0.040452, 0.047320, 0.058019, 0.096814]) * 100.0
flat_esdf_gt_edt = np.array([0.020891, 0.042050, 0.048431, 0.060731, 0.098208]) * 100.0

flat_esdf_time_ours = [88.0, 14.4, 6.4, 2.5, 1.6]
flat_esdf_time_voxblox = [347.9, 44.2, 14.7, 8.1, 4.6]
flat_esdf_time_fiesta = [109.2, 17.5, 7.0, 2.8, 2.1]
flat_esdf_time_edt = [127.9, 19.6, 7.8, 3.4, 2.6]

flat_voxel_size = [5, 10, 15, 20, 25]

# # tsdf error
# l1_tsdf_acc = plt.plot(flat_voxel_size, flat_tsdf_ours, '-o', linewidth = 5, markersize = 20, label='With projective TSDF correction (Ours)')
# l2_tsdf_acc = plt.plot(flat_voxel_size, flat_tsdf_voxblox, '-^', linewidth = 5, markersize = 20, label='Without projective TSDF correction (Voxblox)')

# plt.rcParams.update({'font.size': 35})
# plt.xticks(np.arange(5, 30, step=5), fontsize=30)
# plt.yticks(fontsize=30)
# plt.ylim([0, 10])
# plt.xlabel('Voxel size (cm)', fontsize=40)
# plt.ylabel('TSDF error (cm)', fontsize=40)
# plt.legend()
# plt.show()

# mesh recon error
# l1_mesh_acc = plt.plot(flat_voxel_size, flat_mesh_ours, '-o', linewidth = 5, markersize = 20, label='With projective TSDF correction (Ours)')
# l2_mesh_acc = plt.plot(flat_voxel_size, flat_mesh_voxblox, '-^', linewidth = 5, markersize = 20, label='Without projective TSDF correction (Voxblox)')

# plt.rcParams.update({'font.size': 35})
# plt.xticks(np.arange(5, 30, step=5), fontsize=30)
# plt.yticks(fontsize=30)
# plt.ylim([0, 5])
# plt.xlabel('Voxel size (cm)', fontsize=40)
# plt.ylabel('Mesh reconstruction error (cm)', fontsize=40)
# plt.legend()
# plt.show()

# coverage 
# l1_cov_acc = plt.plot(flat_voxel_size, flat_coverage_ours, '-o', linewidth = 5, markersize = 20, label='With projective TSDF correction (Ours)')
# l2_cov_acc = plt.plot(flat_voxel_size, flat_coverage_voxblox, '-^', linewidth = 5, markersize = 20, label='Without projective TSDF correction (Voxblox)')

# plt.rcParams.update({'font.size': 35})
# plt.xticks(np.arange(5, 30, step=5), fontsize=30)
# plt.yticks(fontsize=30)
# # plt.ylim([0, 5])
# plt.xlabel('Voxel size (cm)', fontsize=40)
# plt.ylabel('Coverage (%)', fontsize=40)
# plt.legend()
# plt.show()

# esdf computation time
## l1_esdf_time = plt.plot(flat_voxel_size, flat_esdf_time_ours, '-o', linewidth = 5, markersize = 20, label='Efficient real Euclidean ESDF mapping (Ours)')
## l2_esdf_time = plt.plot(flat_voxel_size, flat_esdf_time_voxblox, '-^', linewidth = 5, markersize = 20, label='Quasi Euclidean ESDF mapping (Voxblox)')
# l1_esdf_time = plt.plot(flat_voxel_size, flat_esdf_time_ours, '-o', linewidth = 5, markersize = 20, label='Voxfield (Ours)')
# l2_esdf_time = plt.plot(flat_voxel_size, flat_esdf_time_voxblox, '-^', linewidth = 5, markersize = 20, label='Voxblox')
# l3_esdf_time = plt.plot(flat_voxel_size, flat_esdf_time_fiesta, '-v', linewidth = 5, markersize = 20, label='FIESTA')
# l4_esdf_time = plt.plot(flat_voxel_size, flat_esdf_time_edt, '-*', linewidth = 5, markersize = 20, label='EDT')

# plt.rcParams.update({'font.size': 35})
# plt.xticks(np.arange(5, 30, step=5), fontsize=30)
# plt.yticks(fontsize=30)
# plt.yscale('log')
# plt.xlabel('Voxel size (cm)', fontsize=40)
# plt.ylabel('Consumed time per update (ms)', fontsize=40)
# plt.legend()
# plt.show()

# esdf accuracy (occ ref)
# l1_esdf_acc_occ = plt.plot(flat_voxel_size, flat_esdf_occ_ours, '-o', linewidth = 5, markersize = 20, label='Voxfield (Ours)')
# l2_esdf_acc_occ = plt.plot(flat_voxel_size, flat_esdf_occ_voxblox, '-^', linewidth = 5, markersize = 20, label='Voxblox')
# l3_esdf_acc_occ = plt.plot(flat_voxel_size, flat_esdf_occ_fiesta, '-v', linewidth = 5, markersize = 20, label='FIESTA')
# l4_esdf_acc_occ = plt.plot(flat_voxel_size, flat_esdf_occ_edt, '-*', linewidth = 5, markersize = 20, label='EDT')

# plt.rcParams.update({'font.size': 35})
# plt.xticks(np.arange(5, 30, step=5), fontsize=30)
# plt.yticks(fontsize=30)
# # plt.yscale('log')
# # plt.ylim([0, 12])
# plt.xlabel('Voxel size (cm)', fontsize=40)
# plt.ylabel('ESDF error (cm) [Ref: Occupied voxel center] ', fontsize=35)
# plt.legend()
# plt.show()

# esdf accuracy (gt ref)
# l1_esdf_acc_gt = plt.plot(flat_voxel_size, flat_esdf_gt_ours, '-o', linewidth = 5, markersize = 20, label='Voxfield (Ours)')
# l2_esdf_acc_gt = plt.plot(flat_voxel_size, flat_esdf_gt_voxblox, '-^', linewidth = 5, markersize = 20, label='Voxblox')
# l3_esdf_acc_gt = plt.plot(flat_voxel_size, flat_esdf_gt_fiesta, '-v', linewidth = 5, markersize = 20, label='FIESTA')
# l4_esdf_acc_gt = plt.plot(flat_voxel_size, flat_esdf_gt_edt, '-*', linewidth = 5, markersize = 20, label='EDT')

# plt.rcParams.update({'font.size': 35})
# plt.xticks(np.arange(5, 30, step=5), fontsize=30)
# plt.yticks(fontsize=30)
# # plt.yscale('log')
# plt.ylim([0, 12])
# plt.xlabel('Voxel size (cm)', fontsize=40)
# plt.ylabel('ESDF error (cm) [Ref: GT point cloud] ', fontsize=35)
# plt.legend()
# plt.show()


# MaiCity Dataset
mai_tsdf_ours = np.array([0.014968, 0.027714, 0.040541, 0.064733, 0.074458, 0.094721, 0.104881]) * 100.0
mai_tsdf_voxblox = np.array([0.043657, 0.078265, 0.118020, 0.164887, 0.192402, 0.209249, 0.223521]) * 100.0

mai_mesh_ours = np.array([0.036759, 0.058151, 0.068234, 0.101228, 0.114352, 0.124643, 0.140532]) * 100.0
mai_mesh_voxblox = np.array([0.039103, 0.061638, 0.078188, 0.109485, 0.123240, 0.133580, 0.150718]) * 100.0

mai_chamfer_ours = np.array([0.048204, 0.075392, 0.094810, 0.130635, 0.150917, 0.175598, 0.190937]) * 100.0
mai_chamfer_voxblox = np.array([0.051746, 0.077667, 0.100711, 0.133263, 0.153123, 0.177588, 0.196262]) * 100.0

mai_voxel_size = [10, 15, 20, 25, 30, 35, 40]
mai_voxel_size_esdf = [15, 20, 25, 30, 35, 40] # voxel size = 10 is too computational heavily for esdf mapping

mai_esdf_occ_ours = np.array([0.011842, 0.011165, 0.012473, 0.021607, 0.026983, 0.020366]) * 100.0
mai_esdf_occ_ours_2 = np.array([0.048313, 0.079564, 0.086715, 0.101150, 0.140885, 0.129498]) * 100.0
mai_esdf_occ_voxblox = np.array([0.105129, 0.135906, 0.159407, 0.190374, 0.215760, 0.233799]) * 100.0
mai_esdf_occ_fiesta = np.array([0.009990, 0.010689, 0.020770, 0.020486, 0.023124, 0.019748]) * 100.0
mai_esdf_occ_edt = np.array([0.009259, 0.021329, 0.028463, 0.026515, 0.033678, 0.030134]) * 100.0

mai_esdf_gt_ours = np.array([0.061703, 0.092117, 0.117943, 0.140956, 0.162598, 0.182063]) * 100.0
mai_esdf_gt_ours_2 = np.array([0.063367, 0.091992, 0.131013, 0.152049, 0.182574, 0.185861]) * 100.0
mai_esdf_gt_voxblox = np.array([0.118326, 0.175636, 0.217876, 0.261014, 0.283866, 0.300789]) * 100.0
mai_esdf_gt_fiesta = np.array([0.055197, 0.104627, 0.141282, 0.177998, 0.185972, 0.189154]) * 100.0
mai_esdf_gt_edt = np.array([0.057547, 0.109508, 0.144468, 0.180567, 0.191044, 0.197996]) * 100.0

mai_esdf_time_ours = [1295.5, 619.4, 351.6, 206.3, 160.3, 103.3]
mai_esdf_time_voxblox = [2793.1, 1254.9, 685.0, 423.2, 332.8, 283.6]
mai_esdf_time_fiesta = [1910.3, 671.4, 422.1, 284.7, 202.7, 137.9]
mai_esdf_time_edt = [1502.3, 901.7, 449.3, 261.8, 195.7, 121.6]

# tsdf error
# l1_tsdf_acc = plt.plot(mai_voxel_size, mai_tsdf_ours, '-o', linewidth = 5, markersize = 20, label='With projective TSDF correction (Ours)')
# l2_tsdf_acc = plt.plot(mai_voxel_size, mai_tsdf_voxblox, '-^', linewidth = 5, markersize = 20, label='Without projective TSDF correction (Voxblox)')

# plt.rcParams.update({'font.size': 35})
# plt.xticks(np.arange(10, 45, step=5), fontsize=30)
# plt.yticks(fontsize=30)
# plt.ylim([0, 25])
# plt.xlabel('Voxel size (cm)', fontsize=40)
# plt.ylabel('TSDF error (cm)', fontsize=40)
# plt.legend()
# plt.show()

# mesh recon error
# l1_mesh_acc = plt.plot(mai_voxel_size, mai_mesh_ours, '-o', linewidth = 5, markersize = 20, label='With projective TSDF correction (Ours)')
# l2_mesh_acc = plt.plot(mai_voxel_size, mai_mesh_voxblox, '-^', linewidth = 5, markersize = 20, label='Without projective TSDF correction (Voxblox)')

# plt.rcParams.update({'font.size': 35})
# plt.xticks(np.arange(10, 45, step=5), fontsize=30)
# plt.yticks(fontsize=30)
# plt.ylim([0, 16])
# plt.xlabel('Voxel size (cm)', fontsize=40)
# plt.ylabel('Mesh reconstruction error (cm)', fontsize=40)
# plt.legend()
# plt.show()

# mesh recon error (chamfer distance)
# l1_mesh_chamfer = plt.plot(mai_voxel_size, mai_chamfer_ours, '-o', linewidth = 5, markersize = 20, label='With projective TSDF correction (Ours)')
# l2_mesh_chamfer = plt.plot(mai_voxel_size, mai_chamfer_voxblox, '-^', linewidth = 5, markersize = 20, label='Without projective TSDF correction (Voxblox)')
# l3_mesh_chamfer = plt.axhline(y=5.0, linestyle='--', color='g', linewidth = 5, label='Puma') 
# l4_mesh_chamfer = plt.axhline(y=11.0, linestyle='--', color='c', linewidth = 5, label='Surfels') 

# plt.rcParams.update({'font.size': 35})
# plt.xticks(np.arange(10, 45, step=5), fontsize=30)
# plt.yticks(fontsize=30)
# plt.ylim([0, 20])
# plt.xlabel('Voxel size (cm)', fontsize=40)
# plt.ylabel('Chamfer distance (cm)', fontsize=40)
# plt.legend(loc = 'upper left')
# plt.show()

# esdf computation time
# l1_esdf_time = plt.plot(mai_voxel_size_esdf, mai_esdf_time_ours, '-o', linewidth = 5, markersize = 20, label='Voxfield (Ours)')
# l2_esdf_time = plt.plot(mai_voxel_size_esdf, mai_esdf_time_voxblox, '-^', linewidth = 5, markersize = 20, label='Voxblox')
# l3_esdf_time = plt.plot(mai_voxel_size_esdf, mai_esdf_time_fiesta, '-v', linewidth = 5, markersize = 20, label='FIESTA')
# l4_esdf_time = plt.plot(mai_voxel_size_esdf, mai_esdf_time_edt, '-*', linewidth = 5, markersize = 20, label='EDT')

# plt.rcParams.update({'font.size': 35})
# plt.xticks(np.arange(15, 45, step=5), fontsize=30)
# plt.yticks(fontsize=30)
# plt.yscale('log')
# plt.xlabel('Voxel size (cm)', fontsize=40)
# plt.ylabel('Consumed time per update (ms)', fontsize=40)
# plt.legend()
# plt.show()

# esdf accuracy (occ ref)
# l1_esdf_acc_occ = plt.plot(mai_voxel_size_esdf, mai_esdf_occ_ours, '-o', linewidth = 5, markersize = 20, label='Voxfield (Ours)')
# l2_esdf_acc_occ = plt.plot(mai_voxel_size_esdf, mai_esdf_occ_voxblox, '-^', linewidth = 5, markersize = 20, label='Voxblox')
# l3_esdf_acc_occ = plt.plot(mai_voxel_size_esdf, mai_esdf_occ_fiesta, '-v', linewidth = 5, markersize = 20, label='FIESTA')
# l4_esdf_acc_occ = plt.plot(mai_voxel_size_esdf, mai_esdf_occ_edt, '-*', linewidth = 5, markersize = 20, label='EDT')

# plt.rcParams.update({'font.size': 35})
# plt.xticks(np.arange(15, 45, step=5), fontsize=30)
# plt.yticks(fontsize=30)
# # plt.yscale('log')
# # plt.ylim([0, 12])
# plt.xlabel('Voxel size (cm)', fontsize=40)
# plt.ylabel('ESDF error (cm) [Ref: Occupied voxel center] ', fontsize=35)
# plt.legend()
# plt.show()

# esdf accuracy (gt ref)
l1_esdf_acc_gt = plt.plot(mai_voxel_size_esdf, mai_esdf_gt_ours, '-o', linewidth = 5, markersize = 20, label='Voxfield (Ours)')
l2_esdf_acc_gt = plt.plot(mai_voxel_size_esdf, mai_esdf_gt_voxblox, '-^', linewidth = 5, markersize = 20, label='Voxblox')
l3_esdf_acc_gt = plt.plot(mai_voxel_size_esdf, mai_esdf_gt_fiesta, '-v', linewidth = 5, markersize = 20, label='FIESTA')
l4_esdf_acc_gt = plt.plot(mai_voxel_size_esdf, mai_esdf_gt_edt, '-*', linewidth = 5, markersize = 20, label='EDT')

plt.rcParams.update({'font.size': 35})
plt.xticks(np.arange(15, 45, step=5), fontsize=30)
plt.yticks(fontsize=30)
plt.ylim([0, 32])
plt.xlabel('Voxel size (cm)', fontsize=40)
plt.ylabel('ESDF error (cm) [Ref: GT point cloud] ', fontsize=35)
plt.legend()
plt.show()

# l1_cov = plt.plot(voxel_size, flat_coverage_ours)
# l2_cov = plt.plot(voxel_size, flat_coverage_voxblox)

# plt.rcParams.update({'font.size': 35})
# plt.xticks(np.arange(5, 30, step=5), fontsize=30)
# plt.yticks(fontsize=30)
# plt.yscale('log')
# # plt.ylim([0, 15])
# plt.xlabel('Voxel size (cm)', fontsize=40)
# plt.ylabel('Consumed time per update (ms)', fontsize=35)
# plt.legend()
# plt.show()

# labels = ['Flat', 'KITTI']
# without_normal_mae_mean = [0.91, 6.80]
# # without_normal_mae_std = [0.02, 0.07]
# with_normal_mae_mean = [0.76, 4.29]
# # with_normal_mae_std = [0.01, 0.05]

# without_normal_coverage_mean = [66.62, 94.35]
# # without_normal_coverage_std = []
# with_normal_coverage_mean = [76.11, 96.82]
# # with_normal_coverage_std = []

# x = np.arange(len(labels))  # the label locations
# width = 0.25  # the width of the bars

# plt.rcParams.update({'font.size': 24})

# fig_2, ax_2 = plt.subplots()
# rects1 = ax_2.bar(x - width/2, without_normal_mae_mean, width, label='Without correction')
# rects2 = ax_2.bar(x + width/2, with_normal_mae_mean, width, label='With correction (Ours)')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax_2.set_ylabel('Mean reconstruction error (cm)')
# ax_2.set_xticks(x)
# ax_2.set_xticklabels(labels)

# ax_2.legend()

# fig_2.set_size_inches(9.5, 10.5)
# fig_2.tight_layout()


# fig_3, ax_3 = plt.subplots()
# rects1 = ax_3.bar(x - width/2, without_normal_coverage_mean, width, label='Without correction')
# rects2 = ax_3.bar(x + width/2, with_normal_coverage_mean, width, label='With correction (Ours)')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax_3.set_ylabel('Coverage (%)')
# ax_3.set_xticks(x)
# ax_3.set_xticklabels(labels)

# ax_3.legend()

# fig_3.set_size_inches(9.5, 10.5)
# fig_3.tight_layout()

# plt.show()