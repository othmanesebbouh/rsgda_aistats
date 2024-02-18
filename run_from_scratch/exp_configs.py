from haven import haven_utils as hu

run_list = [0, 1, 2, 3, 4]

EXP_GROUPS = {'fig1_proba': hu.cartesian_exp_group({"dataset": ["mnist2d"],
                                                    "model": ["ConvElu"],
                                                    "loss_func": ["softmax_loss"],
                                                    "opt": [{'name': "sgd"}],
                                                    "lr_d": [0.2, 0.25, 0.3],
                                                    "lr_a": [10],
                                                    "loop_size": [None],
                                                    "p": [0.5, 0.25],
                                                    "gamma": [1.3],
                                                    "acc_func": ["softmax_accuracy"],
                                                    "batch_size": [128],
                                                    "max_epoch": [200],
                                                    "phi_epochs_before_check": [10],
                                                    "grad_phi_every": [20],
                                                    "max_epoch_phi": [5],
                                                    "lr_compute_phi": [50],
                                                    "tol_compute_phi": [1e-4],
                                                    "runs": run_list}),
              'fig1_loop': hu.cartesian_exp_group({"dataset": ["mnist2d"],
                                                   "model": ["ConvElu"],
                                                   "loss_func": ["softmax_loss"],
                                                   "opt": [{'name': "sgd"}],
                                                   "lr_d": [0.2, 0.25, 0.3],
                                                   "lr_a": [10],
                                                   "loop_size": [1, 3],
                                                   "p": [None],
                                                   "gamma": [1.3],
                                                   "acc_func": ["softmax_accuracy"],
                                                   "batch_size": [128],
                                                   "max_epoch": [200],
                                                   "phi_epochs_before_check": [10],
                                                   "grad_phi_every": [20],
                                                   "max_epoch_phi": [5],
                                                   "lr_compute_phi": [50],
                                                   "tol_compute_phi": [1e-4],
                                                   "runs": run_list}),
              'fig2': hu.cartesian_exp_group({"dataset": ["mnist2d"],
                                              "model": ["ConvElu"],
                                              "loss_func": ["softmax_loss"],
                                              "opt": [{'name': "sgd"}],
                                              "lr_d": [0.2, 0.25, 0.3],
                                              "lr_a": [10],
                                              "loop_size": [None],
                                              "p": [0.1, 0.25, 0.5, 0.9],
                                              "gamma": [1.3],
                                              "acc_func": ["softmax_accuracy"],
                                              "batch_size": [128],
                                              "max_epoch": [200],
                                              "phi_epochs_before_check": [10],
                                              "grad_phi_every": [20],
                                              "max_epoch_phi": [5],
                                              "lr_compute_phi": [50],
                                              "tol_compute_phi": [1e-4],
                                              "runs": run_list})}

