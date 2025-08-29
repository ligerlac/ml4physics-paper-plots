from pathlib import Path
from typing import List, Callable, Tuple, Union, Dict

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import numpy.typing as npt

from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec
from matplotlib.lines import Line2D
# from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

from utils import get_fractions_above_threshold, get_rounded_str

# Color scheme from https://github.com/mpetroff/accessible-color-cycles/tree/master (recommended by root team)
# ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]  # 6 colors
# ["#1845fb", "#ff5e02", "#c91f16", "#c849a9", "#adad7d", "#86c8dd", "#578dff", "#656364"]  # 8 colors
# ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]  # 10 colors


class Draw:
    def __init__(self, output_dir: Path = Path("plots"), interactive: bool = False, output_format: str = "png"):
        self.output_dir = output_dir
        self.interactive = interactive
        self.output_format = output_format
        self.cmap = ["green", "red", "blue", "orange", "purple", "brown"]
        self.model_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        self.process_color_dict = {
            "ZB": "#1845fb",
            "ZB-masked": "#578dff",
            "SingleNeutrino": "#656364",
            "GluGluHToGG": "#ff5e02",
            "TT": "#c91f16",
            "VBFHto2B": "#adad7d",
            "HTo2LongLivedTo4b": "#c849a9",
            "SUEP": "#86c8dd",
        }
        self.model_color_dict = dict()
        self.label_dict = {
            "ZB-masked": "ZB (npv > 10)",
            "SingleNeutrino": "Simulated ZB",
        }
        self.cms_text = "Open Data"
        self.lumi_text = r'2017 (13 TeV)'
        hep.style.use("CMS")

    def _parse_name(self, name: str) -> str:
        return name.replace(" ", "-").lower()
    
    def _get_process_color(self, label: str) -> str:
        return self.process_color_dict.get(label, None)
        # return self.process_color_dict.get(label, next(plt.gca()._get_lines.prop_cycler)['color'])

    def _get_model_color(self, label: str) -> str:
        return self.model_color_dict.get(label, None)
        # return self.model_color_dict.get(label, next(plt.gca()._get_lines.prop_cycler)['color'])
    
    def _get_label(self, label: str) -> str:
        return self.label_dict.get(label, label)

    def _save_fig(self, name: str) -> None:
        plt.savefig(
            f"{self.output_dir}/{self._parse_name(name)}.{self.output_format}", bbox_inches="tight", format=self.output_format
        )
        if self.interactive:
            plt.show()
        plt.close()

    def plot_loss_history(
        self, training_loss: npt.NDArray, validation_loss: npt.NDArray, name: str
    ):
        plt.plot(np.arange(1, len(training_loss) + 1), training_loss, label="Training")
        plt.plot(
            np.arange(1, len(validation_loss) + 1), validation_loss, label="Validation"
        )
        plt.legend(loc="upper right")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        self._save_fig(name)

    def plot_loss_histories(
        self, loss_dict: dict[str, (npt.NDArray, npt.NDArray)], name: str
    ):
        for model_name, (train_loss, val_loss) in loss_dict.items():
            # c = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.plot(np.arange(1, len(train_loss) + 1), train_loss, color=c, label=f"{model_name} (Training)")
            plt.plot(np.arange(1, len(val_loss) + 1), val_loss, color=c, ls=":", label=f"{model_name} (Validation)")
        plt.legend(loc="upper right")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        self._save_fig(name)

    def plot_regional_deposits(
        self, deposits: npt.NDArray, mean: float, name: str, is_data: bool = False,
    ):
        im = plt.imshow(
            deposits.reshape(18, 14), vmin=0, vmax=deposits.max(), cmap="Purples"
        )
        ax = plt.gca()
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(r"Calorimeter E$_T$ deposit (GeV)")
        plt.xticks(np.arange(14), labels=np.arange(4, 18))
        plt.yticks(
            np.arange(18),
            labels=np.arange(18)[::-1],
            rotation=90,
            va="center",
        )
        plt.xlabel(r"i$\eta$")
        plt.ylabel(r"i$\phi$")

        if is_data:
            hep.cms.text('Preliminary', ax=ax, pad=0.05)
        else:
            hep.cms.text('Simulation Preliminary', ax=ax, pad=0.05)

        # verbose title
        hep.cms.lumitext(rf'$<E_T> = {mean: .2f}$; {name} (13 TeV)', ax=ax)

        # short title
        # hep.cms.lumitext('2023 (13 TeV)', ax=ax)

        self._save_fig(f'profiling-mean-deposits-{name}')

    def plot_spacial_deposits_distribution(
        self, deposits: List[npt.NDArray], labels: List[str], name: str, apply_weights: bool = False
    ):
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        for deposit, label in zip(deposits, labels):
            bins = np.argwhere(deposit)
            phi, eta = bins[:, 1], bins[:, 2]
            if apply_weights:
                weights = deposit[np.nonzero(deposit)]
            else:
                weights = np.ones(phi.shape)
            ax1.hist(
                eta + 4,
                weights=weights,
                density=True,
                facecolor=None,
                bins=np.arange(4, 19),
                label=label,
                histtype="step"
            )
            ax2.hist(
                phi,
                weights=weights,
                density=True,
                facecolor=None,
                bins=np.arange(19),
                label=label,
                histtype="step",
            )
        ax1.set_ylabel("a.u.")
        ax1.set_xlabel(r"i$\eta$")
        ax2.set_xlabel(r"i$\phi$")
        plt.legend(loc="best")
        self._save_fig(f'profiling-spacial-{name}')

    def plot_deposits_distribution(
        self, deposits: List[npt.NDArray], labels: List[str], name: str,
    ):
        for deposit, label in zip(deposits, labels):
            plt.hist(
                deposit.reshape((-1)),
                bins=100,
                range=(0, 1024),
                density=1,
                label=label,
                log=True,
                histtype="step",
            )
        plt.xlabel(r"E$_T$")
        plt.legend(loc="best")
        self._save_fig(f'profiling-deposits-{name}')

    def plot_cell_means(
        self, deposits: npt.NDArray, name: str
    ):
        d = np.squeeze(deposits, axis=-1)
        means = np.mean(d, axis=0)
        stds = np.std(d, axis=0)
        sems = stds / np.sqrt(d.shape[0])

        x = np.arange(36)

        for eta in range(7):
            m = np.concatenate([means[:, eta], means[:, 13-eta]])
            s = np.concatenate([sems[:, eta], sems[:, 13-eta]])
            l = m - s
            u = m + s
            plt.plot(x, m)
            plt.xlabel(r"i$\phi$")
            plt.ylabel(r"Mean E$_T$ deposit (GeV)")
            plt.fill_between(x, l, u, alpha=0.1, label=f'$i\eta={eta}$')
        plt.axvline(x=17.5, ls=':', color='grey', alpha=0.5)
        plt.gca().set_xticks(np.arange(36)[::2], np.concatenate([np.arange(18), np.arange(18)])[::2])
        plt.legend(ncols=2)
        self._save_fig(f'profiling-deposits-{name}')

    def plot_cell_dists(
        self, deposits: npt.NDArray, name: str
    ):
        bins = np.arange(20) - 0.5
        d = np.squeeze(deposits, axis=-1)
        for eta in range(1):
            for phi in range(2):
                ets = d[:, phi, eta]
                plt.hist(ets, bins, alpha=0.5, label=f'i\phi = {phi}')
            plt.legend()
            self._save_fig(f'et-dist-region-eta-{eta}')

    def plot_reconstruction_results(
        self,
        deposits_in: npt.NDArray,
        deposits_out: npt.NDArray,
        loss: float,
        name: str,
        is_data: bool = False,
    ):
        fig, (ax1, ax2, ax3, cax) = plt.subplots(
            ncols=4, figsize=(15, 10), gridspec_kw={"width_ratios": [1, 1, 1, 0.05]}
        )
        max_deposit = max(deposits_in.max(), deposits_out.max())

        if is_data:
            hep.cms.text('Preliminary', ax=ax1, fontsize=18)
        else:
            hep.cms.text('Simulation Preliminary', ax=ax1, fontsize=18)
        hep.cms.lumitext('2023 (13 TeV)', ax=ax3, fontsize=18)

        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax1.set_title("Original", fontsize=18, y=-0.1)
        ax1.imshow(
            deposits_in.reshape(18, 14), vmin=0, vmax=max_deposit, cmap="Purples"
        )
        
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax2.set_title("Reconstructed", fontsize=18, y=-0.1)
        ax2.imshow(
            deposits_out.reshape(18, 14), vmin=0, vmax=max_deposit, cmap="Purples"
        )

        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        ax3.set_title(rf"|$\Delta$|, MSE: {loss: .2f}", fontsize=18, y=-0.1)

        im = ax3.imshow(
            np.abs(deposits_in - deposits_out).reshape(18, 14),
            vmin=0,
            vmax=max_deposit,
            cmap="Purples",
        )

        ip = InsetPosition(ax3, [1.05, 0, 0.05, 1])
        cax.set_axes_locator(ip)
        fig.colorbar(im, cax=cax, ax=[ax1, ax2, ax3]).set_label(
            label=r"Calorimeter E$_T$ deposit (GeV)", fontsize=18
        )
        self._save_fig(name)

    def plot_individual_image(
        self,
        deposits: npt.NDArray,
        name: str,
    ):

        im = plt.imshow(
            deposits.reshape(18, 14), vmin=0, vmax=deposits.max(), cmap="Purples"
        )

        ax = plt.gca()
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_ticks([])
        cbar.ax.set_ylabel(r"Energy deposit")
        plt.xticks([])
        plt.yticks([])
        ax.tick_params(length=0, width=0)
        plt.xlabel(r"i$\eta$")
        plt.ylabel(r"i$\phi$")
 
        self._save_fig(name)

    def plot_phi_shift_variance(
        self, losses: List[float], name: str
    ):
        x = np.arange(len(losses))
        loss_means = np.mean(losses, axis=1)
        plt.plot(x, loss_means)
        loss_stds =  np.std(losses, axis=1)
        lower = loss_means - loss_stds / 2
        upper = loss_means + loss_stds / 2
        plt.fill_between(x, lower, upper, alpha=0.1)
        plt.xlabel(r"Shift [$\Delta$ i$\phi$]")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel(r"$\Delta_{rel} (MSE)$")
        plt.axvline(x=0, color='grey', linestyle=':', label='Original')
        plt.axvline(x=18, color='grey', linestyle=':')
        plt.axhline(y=loss_means[0], color='grey', linestyle=':')
        plt.legend()
        self._save_fig(name)


    def plot_anomaly_score_distribution(
        self,
        scores: List[npt.NDArray],
        labels: List[str],
        name: str = "anomaly_score_distribution",
        xlabel: str = r"Anomaly Score",
        left_legend_col: tuple[int] = None,
        bins: list = range(0, 258, 2),
        xticks: list = None,
        y_max: float = 4,
        show_mean: bool = False
    ):
        """
        @param scores: List of numpy arrays containing the anomaly scores for each label.
        @param labels: List of labels corresponding to the scores.
        @param name: Name of the plot to be saved.
        @param xlabel: Label for the x-axis.
        @param left_legend_col: indices of entries for the left col of the legend (single legend if None)
        """
        plt.figure(figsize=(9.5, 9.5))
        hs = {}
        for score, label in zip(scores, labels):
            label_ = self._get_label(label)
            if show_mean:
                label_ = f"{label_} ({get_rounded_str(np.mean(score))})",
            _,_, hs[label] = plt.hist(
                score.reshape((-1)),
                bins=bins,
                density=1,
                label=label_,
                log=True,
                histtype="step",
                linewidth=2,
                color=self._get_process_color(label),
            )

        plt.xlabel(xlabel)
        plt.ylabel("a.u.")
        ax = plt.gca()

        if left_legend_col is not None:
            # # sort labels by length
            # sorted_labels = sorted(labels, key=len)
            # left_keys = sorted_labels[:len(sorted_labels)//2-1]
            # right_keys = sorted_labels[len(sorted_labels)//2-1:]
            # left_keys = [self._get_label(labels[i]) for i in left_legend_col]
            # right_keys = [self._get_label(l) for l in labels if l not in left_keys]
            left_keys = [labels[i] for i in left_legend_col]
            right_keys = [l for l in labels if l not in left_keys]
            print(f"Left legend keyzs: {left_keys}, Right legend keys: {right_keys}")
            l2 = ax.legend(handles=[hs[k][0] for k in left_keys], loc="upper left", bbox_to_anchor=(0.0, 1.0))
            l1 = ax.legend(handles=[hs[k][0] for k in right_keys], loc="upper right", bbox_to_anchor=(1.0, 1.0))
            ax.add_artist(l1)
            ax.add_artist(l2)
        else:
            plt.legend(loc="upper right")
        ax.set_ylim(0.0000005, y_max)

        if xticks is not None:
            ax.set_xticks(xticks)
        # else:
        #     ax.set_xticks(range(0, max(bins)+1, (max(bins)+1)//8))    
                
        hep.cms.text(self.cms_text, loc=0)
        hep.cms.lumitext(self.lumi_text)

        plt.tight_layout()
        
        self._save_fig(name)

    
    def plot_roc_curves(
        self,
        roc_dict: Dict[str, Tuple[npt.NDArray, npt.NDArray]],
        roc_dict_alt: Dict[str, Tuple[npt.NDArray, npt.NDArray]] = None,
        alt_legend: Tuple[str, str] = None,
        name: str = "roc_curve",
        xlabel: str = "Trigger Rate [kHz]",
        ylabel: str = "Signal Efficiency",
        xrange: Tuple[float, float] = (0.2, 100),
        yrange: Tuple[float, float] = (0.001, 1.),
        show_auc: bool = True,
        working_points: Dict[str, float] = None,
        fpr_scale_factor: float = 28610,
        xlog: bool = True,
        ylog: bool = True,
        figsize: Tuple[int, int] = (9.5, 9.5),
    ):
        """
        Plot ROC curves for multiple processes.

        @param roc_dict: Dictionary where keys are process labels and values are tuples of (fpr, tpr).
        @param roc_dict_alt: Optional dictionary for alternative ROC curves (dashed lines).
        @param alt_legend: Tuple containing the legnd title and the labels for second legend (title, (label1, label2)).
        @param name: Name of the plot to be saved.
        @param xlabel: Label for the x-axis.
        @param ylabel: Label for the y-axis.
        @param xrange: Range for the x-axis.
        @param yrange: Range for the y-axis.
        @param show_auc: Whether to display the AUC in the legend.
        @param working_points: Dictionary of working points with their trigger rates drawn as vertival lines.
        """
        plt.figure(figsize=figsize)

        for label, (fpr, tpr) in roc_dict.items():
            label_ = self._get_label(label)
            if show_auc:
                auc_ = auc(fpr, tpr)
                label_ = f"{label_} (AUC ={auc_: .2f})"
            plt.plot(
                fpr * fpr_scale_factor,
                tpr,
                linestyle="-",
                lw=2,
                color=self._get_process_color(label),
                alpha=0.8,
                label=label_,
            )
        
        if roc_dict_alt is not None:
            lines = plt.gca().get_lines()
            for i, (label, (fpr_alt, tpr_alt)) in enumerate(roc_dict_alt.items()):
                label_ = self._get_label(label)
                if show_auc:
                    auc_ = auc(fpr_alt, tpr_alt)
                    old_label = lines[i].get_label()
                    lines[i].set_label(old_label.replace(")", f" [{auc_:.2f}])"))
                plt.plot(
                    fpr_alt * fpr_scale_factor,
                    tpr_alt,
                    linestyle="--",
                    lw=2,
                    color=self._get_process_color(label),
                    alpha=0.5,
                )

        legend1 = plt.legend(loc='upper left')
        plt.gca().add_artist(legend1)

        if alt_legend is not None:
            legend_elements = [
                Line2D([0], [0], color='grey', linewidth=2, linestyle='-', 
                    label=self._get_label(alt_legend[1][0])),
                Line2D([0], [0], color='grey', linewidth=2, linestyle='--', 
                    label=self._get_label(alt_legend[1][1]))
            ]
            plt.legend(
                handles=legend_elements,
                title=alt_legend[0],
                title_fontsize=20,
                loc='lower right'
            )

        plt.xlim(xrange)
        plt.ylim(yrange)
        if xlog:
            plt.xscale("log")
        if ylog:
            plt.yscale("log")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if working_points is not None:
            for rate, wp in working_points:
                tprs = [tpr for fpr, tpr in roc_dict.values()]
                tprs = [tpr[np.argmin(np.abs(fpr * 28610 - rate))] for fpr, tpr in roc_dict.values()]
                best_tpr = max(tprs)
                # the best tpr for the given rate
                # best_tpr = max(tpr for f, t in roc_dict.values() if fpr[np.argmin(np.abs(fpr * 28610 - rate))] == rate)
                print(f"Working point {wp} at rate {rate} kHz with TPR {best_tpr:.2f}")
                plt.plot([rate, rate], [0, best_tpr], linestyle='--', color='grey', linewidth=1)
                # plt.axvline(
                #     x=rate,
                #     ymax=best_tpr,
                #     # ymax=0.65,
                #     # ymin=-0.1,
                #     # clip_on=False,
                #     linestyle="--",
                #     lw=1,
                #     color="grey",
                # )
                # plt.text(
                #     x=rate,
                #     # y=-0.05,  # Offset below x-axis
                #     y=0.65,
                #     # s=f"{wp}\n({rate} kHz)",
                #     s=wp,
                #     ha='center',
                #     va='top',
                #     # fontsize=14,
                #     color='grey',
                #     transform=plt.gca().get_xaxis_transform()  # x in data coords, y in axes coords
                # )            

            tick_positions = [x[0] for x in working_points if x[1] != ""] + [10, 100]
            tick_labels = [f"{x[0]}\n{x[1]}" for x in working_points if x[1] != ""] + ['10', '100']
            plt.xticks(tick_positions, tick_labels)

        hep.cms.text(self.cms_text, loc=0)
        hep.cms.lumitext(self.lumi_text)

        self._save_fig(name)


    def plot_roc_curve(
        self,
        y_trues: List[npt.NDArray],
        y_preds: List[npt.NDArray],
        labels: List[str],
        name: str = "roc_curve",
        y_preds_baseline: List[npt.NDArray] = None,
        cv: int = 3,
        xlabel: str = "Trigger Rate [kHz]",
        ylabel: str = "Signal Efficiency",
        calc_error: bool = False,
        main_name: str = "CICADA",
        baseline_name: str = "Baseline",
    ):

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        # for i, (y_true, y_pred, label, color) in enumerate(zip(
        #     y_trues, y_preds, labels, self.cmap
        # )):
        for i, (y_true, y_pred, label) in enumerate(zip(y_trues, y_preds, labels)):
            fpr, tpr, _ = roc_curve(y_true, y_pred, drop_intermediate=False)

            if calc_error:
                aucs = []
                for _, indices in skf.split(y_pred, y_true):
                    fpr_, tpr_, _ = roc_curve(y_true[indices], y_pred[indices])
                    aucs.append(auc(fpr_, tpr_))
                std_auc = np.std(aucs)
                extended_label = rf"{label} (AUC ={np.mean(aucs): .2f} $\pm$ {std_auc: .2f})"
            else:
                roc_auc = auc(fpr, tpr)
                extended_label = rf"{label} (AUC ={roc_auc: .2f})"

            plt.plot(
                fpr * 28610,
                tpr,
                linestyle="-",
                lw=2,
                color=self._get_process_color(label),
                alpha=0.8,
                label=extended_label,
            )

            if y_preds_baseline is not None:
                fpr_base, tpr_base, _ = roc_curve(y_true, y_preds_baseline[i])
                plt.plot(
                    fpr_base * 28610,
                    tpr_base,
                    linestyle="--",
                    lw=1.0,
                    color=self._get_process_color(label),
                    alpha=0.5,
                )

        # grey vertical line at envisioned trigger rate
        # plt.plot(
        #     [3, 3],
        #     [0, 1],
        #     linestyle="--",
        #     lw=1,
        #     color="black",
        #     label="3 kHz",
        # )

        plt.xlim([0.2, 100])
        plt.ylim([0.001, 1.1])
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        first_legend = plt.legend(loc='lower right')
        plt.gca().add_artist(first_legend)
    
        # Add second legend (styles in grey)
        style_handles = [
            Line2D([0], [0], color='grey', linestyle='-', lw=2, alpha=0.6),
            Line2D([0], [0], color='grey', linestyle='--', lw=2, alpha=0.6)
        ]
        style_labels = [main_name, baseline_name]

        if y_preds_baseline is not None:
            plt.legend(style_handles, style_labels, loc='upper left')
    
        hep.cms.text('Preliminary', loc=0)
        hep.cms.lumitext(r'2024 (13.6 TeV)')

        self._save_fig(name)


    def plot_compilation_error(
        self, scores_keras: npt.NDArray, scores_hls4ml: npt.NDArray, name: str
    ):
        plt.scatter(scores_keras, np.abs(scores_keras - scores_hls4ml), s=1)
        plt.xlabel("Anomaly Score, $S$")
        plt.ylabel("Error, $|S_{Keras} - S_{hls4ml}|$")
        self._save_fig(f'compilation-error-{name}')

    def plot_compilation_error_distribution(
        self, scores_keras: npt.NDArray, scores_hls4ml: npt.NDArray, name: str
    ):
        plt.hist(scores_keras - scores_hls4ml, fc="none", histtype="step", bins=100)
        plt.xlabel("Error, $S_{Keras} - S_{hls4ml}$")
        plt.ylabel("Number of samples")
        plt.yscale("log")
        self._save_fig(f'compilation-error-dist-{name}')

    def plot_cpp_model(self, hls_model, name: str):
        import hls4ml
        hls4ml.utils.plot_model(
            hls_model,
            show_shapes=True,
            show_precision=True,
            to_file=f"{self.output_dir}/cpp-model-{self._parse_name(name)}.png",
        )

    def plot_roc_curve_comparison(
        self, scores_keras: dict, scores_hls4ml: npt.NDArray, name: str
    ):
        fpr_model: list = []
        tpr_model: list = []

        scores_keras_normal = scores_keras["Background"]
        scores_hls4ml_normal = scores_hls4ml["Background"]

        for dataset_name, color in zip(list(scores_keras.keys())[:-1], self.cmap):
            scores_keras_anomaly = scores_keras[dataset_name]
            scores_hls4ml_anomaly = scores_hls4ml[dataset_name]

            y_true = np.append(
                np.zeros(len(scores_keras_normal)), np.ones(len(scores_hls4ml_anomaly))
            )
            y_score_keras = np.append(scores_keras_normal, scores_keras_anomaly)
            y_score_hls = np.append(scores_hls4ml_normal, scores_hls4ml_anomaly)

            for y_scores, model, ls in zip(
                [y_score_keras, y_score_hls], ["Keras", "hls4ml"], ["-", "--"]
            ):
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                plt.plot(
                    fpr * 28.61,
                    tpr,
                    linestyle=ls,
                    color=color,
                    label="{0}: {1}, AUC = {2:.4f}".format(
                        model, dataset_name, auc(fpr, tpr)
                    ),
                )

        plt.plot(
            [0.003, 0.003],
            [0, 1],
            linestyle="--",
            color="black",
            label="3 kHz trigger rate",
        )
        plt.xlim([0.0002861, 28.61])
        plt.ylim([0.01, 1.0])
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Trigger Rate (MHz)")
        plt.ylabel("Signal Efficiency")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        self._save_fig(f'compilation-roc-{name}')

    def plot_output_reference(self):
        with open("misc/output-reference.txt") as f:
            data = f.read()
        data = np.array([row.split(",") for row in data.split("\n")[:-1]]).astype(
            np.int8
        )
        data = np.flipud(data) - 1
        legend_elements = [
            Patch(
                facecolor=self.cmap[0],
                edgecolor=self.cmap[0],
                label="Anomaly Detection, Integer Part",
            ),
            Patch(
                facecolor=self.cmap[1],
                edgecolor=self.cmap[1],
                label="Anomaly Detection, Decimal Part",
            ),
            Patch(
                facecolor=self.cmap[2], edgecolor=self.cmap[2], label="Heavy Ion Bit"
            ),
            Patch(facecolor=self.cmap[3], edgecolor=self.cmap[3], label="Reserved"),
        ]
        plt.figure(figsize=(25, 5))
        plt.pcolor(
            data, edgecolors="black", alpha=0.6, cmap=ListedColormap(self.cmap[:4])
        )
        plt.xticks([])
        plt.yticks([])
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                plt.text(
                    x + 0.5,
                    y + 0.5,
                    abs(y * 32 + x - 191),
                    horizontalalignment="center",
                    fontsize=16,
                    verticalalignment="center",
                )
        plt.legend(
            handles=legend_elements,
            bbox_to_anchor=(0, 0),
            loc="upper left",
            frameon=False,
            ncol=4,
            borderaxespad=0,
        )
        self._save_fig('ugt-link-reference')

    def plot_results_supervised(
        self, grid: npt.NDArray, models: list[str], datasets: list[str], name: str
    ):
        plt.imshow(grid, alpha=0.7, cmap="RdYlGn")
        plt.xticks(
            np.arange(len(models)),
            labels=models,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        plt.yticks(np.arange(len(datasets)), labels=datasets)
        for i in range(len(datasets)):
            for j in range(len(models)):
                text = plt.text(
                    j,
                    i,
                    "{0:.3f}".format(grid[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                    size=16,
                )
        self._save_fig(f'supervised-{name}')

    def make_equivariance_plot(
        self,
        image: npt.NDArray,
        f: Callable[npt.NDArray, npt.NDArray],  # symmetry transformation
        g: Callable[npt.NDArray, npt.NDArray],  # mapping of the model
        name: str
    ):

        fig, axs = plt.subplots(
            nrows=2, ncols=4, figsize=(15, 10), gridspec_kw={"width_ratios": [1, 1, 1, 0.05]}
        )
        max_deposit = image.max()
        xmax, ymax, _ = image.shape

        mse_g_1 = float(np.mean((g(image) - image)**2))
        mse_gf_f = float(np.mean((g(f(image)) - f(image))**2))
        mse_gf_fg = float(np.mean((g(f(image)) - f(g(image)))**2))

        axs[0, 0].imshow(image, vmin=0, vmax=max_deposit, cmap="Purples")
        axs[0, 1].imshow(f(image), vmin=0, vmax=max_deposit, cmap="Purples")
        im = axs[0, 2].imshow(g(f(image)), vmin=0, vmax=max_deposit, cmap="Purples")
        ip = InsetPosition(axs[0][2], [1.05, 0, 0.05, 1])
        axs[0][3].set_axes_locator(ip)
        fig.colorbar(im, cax=axs[0][3], ax=axs[0][:-1]).set_label(
            label=r"Calorimeter E$_T$ deposit (GeV)", fontsize=18
        )

        axs[1, 0].imshow(image, vmin=0, vmax=max_deposit, cmap="Purples")
        axs[1, 1].imshow(g(image), vmin=0, vmax=max_deposit, cmap="Purples")
        im = axs[1, 2].imshow(f(g(image)), vmin=0, vmax=max_deposit, cmap="Purples")
        ip = InsetPosition(axs[1][2], [1.05, 0, 0.05, 1])
        axs[1][3].set_axes_locator(ip)
        fig.colorbar(im, cax=axs[1][3], ax=axs[1][:-1]).set_label(
            label=r"Calorimeter E$_T$ deposit (GeV)", fontsize=18
        )

        axs[0, 0].annotate('', xy=(1.4, 0.5), xycoords='axes fraction', 
                           xytext=(1.0, 0.5), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', arrowstyle='->'))
        axs[0, 0].text(xmax-3.5, ymax/2+1, 'trans', fontsize=18)

        axs[0, 1].annotate('', xy=(1.4, 0.5), xycoords='axes fraction', 
                           xytext=(1.0, 0.5), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', arrowstyle='->'))
        axs[0, 1].text(xmax-3.5, ymax/2+1, 'pred', fontsize=18)
        axs[0, 1].text(xmax-4, ymax/2+3, rf"MSE: {mse_gf_f:.1f}", fontsize=16)

        axs[1, 0].annotate('', xy=(1.4, 0.5), xycoords='axes fraction', 
                           xytext=(1.0, 0.5), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', arrowstyle='->'))
        axs[1, 0].text(xmax-3.5, ymax/2+1, 'pred', fontsize=18)
        axs[1, 0].text(xmax-4, ymax/2+3, rf"MSE: {mse_g_1:.1f}", fontsize=16)

        axs[1, 1].annotate('', xy=(1.4, 0.5), xycoords='axes fraction', 
                           xytext=(1.0, 0.5), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', arrowstyle='->'))
        axs[1, 1].text(xmax-3.5, ymax/2+1, 'trans', fontsize=18)

        axs[0, 2].annotate('', xy=(0.5, -0.2), xycoords='axes fraction', 
                           xytext=(0.5, 0), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', arrowstyle='<->'))
        axs[0, 2].text(xmax/2-1.5, ymax+6, rf"MSE: {mse_gf_fg:.2f}", fontsize=16)

        for row in axs:
            for ax in row[:-1]:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        self._save_fig(name)


    def make_unrolling_plot(self, deposits: npt.NDArray, name: str, make_animation=False):

        def _draw_angle_arcs(ax_, theta_=65, phi_=45):
            # Draw beam  axis
            ax_.plot((-0.2, 1.2), (0, 0), (0, 0), color='black')
            # ax_.text(0, 0, 0, 'Beam Axis', color='black', fontsize=12)

            # Define a point on the surface of the cylinder
            # Choose a specific angle (for example, 45 degrees) and height (for example, half the height)
            my_phi = np.deg2rad(phi_)
            my_theta = np.deg2rad(theta_)
            surface_height = n_eta / 2

            # Coordinates of the point on the surface
            r = 1 / np.sin(my_theta)
            x_surface = r * np.sin(my_theta) * np.cos(my_phi)
            y_surface = r * np.sin(my_theta) * np.sin(my_phi)
            z_surface = r * np.cos(my_theta)

            # Draw red arrows
            ax_.quiver(0.5, 0, 0, z_surface, x_surface, y_surface, color='r', arrow_length_ratio=0.1)
            ax_.quiver(0.5+z_surface, 0, 0, 0, x_surface, y_surface, color='r', arrow_length_ratio=0.1, linestyle=':')

            # Draw labeled archs for the angles
            angle_radius = 0.3
            phis = np.linspace(my_phi, np.pi, 100)
            if abs(my_phi) > np.pi:
                phis = np.linspace(np.pi, my_phi+2*np.pi, 100)
            phi_arc_x = angle_radius * np.cos(phis)
            phi_arc_y = angle_radius * np.sin(phis)
            phi_arc_z = np.ones_like(phi_arc_x) * r * np.cos(my_theta) + 0.5
            ax_.plot(phi_arc_z, phi_arc_x, phi_arc_y, color='blue', linewidth=2, label='Polar Angle')
            ax_.text(phi_arc_z[0], phi_arc_x[0], phi_arc_y[0], r"$\phi$", color='blue', fontsize=18)

            theta_arc_grid = np.linspace(0, my_theta, 100)
            theta_arc_x = angle_radius * np.sin(theta_arc_grid) * np.cos(my_phi)
            theta_arc_y = angle_radius * np.sin(theta_arc_grid) * np.sin(my_phi)
            theta_arc_z = angle_radius * np.cos(theta_arc_grid) + 0.5
            ax_.plot(theta_arc_z, theta_arc_x, theta_arc_y, color='blue', linewidth=2, label='Polar Angle')
            ax_.text(theta_arc_z[50], theta_arc_x[50], theta_arc_y[50], r"$\theta$", color='blue', fontsize=18)

        def _draw_red_square(ax_, i_eta, i_phi):
            for r in ax_.patches:
                r.remove()
            rect = Rectangle((i_eta-0.5, i_phi-0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
            ax_.add_patch(rect)
            
        # Define the grid dimensions (cylinder height and unrolled grid size)
        n_phi, n_eta = deposits.shape

        # Create cylindrical coordinates for the 3D plot
        theta = np.linspace(0, 2 * np.pi, n_phi)  # Circumference angles
        z_cyl = np.linspace(0, 1, n_eta)  # Height values
        Z_cyl, Theta = np.meshgrid(z_cyl, theta)

        # Convert cylindrical coordinates to Cartesian for plotting
        X_cyl = np.cos(Theta)  # X-coordinates on cylinder's surface (for circular shape)
        Y_cyl = np.sin(Theta)  # Y-coordinates on cylinder's surface

        spec = gridspec.GridSpec(
            ncols=2, nrows=1,
            width_ratios=[2, 1.8], wspace=0.1,
        )

        fig = plt.figure(figsize=(12, 6))
        fig.subplots_adjust(left=0, bottom=0.05, right=1, top=0.95, wspace=None, hspace=0.1)

        ax1 = fig.add_subplot(spec[0], projection='3d')
        ax2 = fig.add_subplot(spec[1])

        # Initial plot for the cylinder
        surf = ax1.plot_surface(Z_cyl, X_cyl, Y_cyl, facecolors=plt.cm.Purples(deposits), 
                                rstride=1, cstride=1, alpha=0.6, linewidth=0)
        ax1.set_title('CMS Calorimeter (Schematic)', fontsize=18)
        ax1.set_axis_off() 
        ax1.view_init(elev=30, azim=-60)
        ax1.dist = 8

        _draw_angle_arcs(ax1, phi_=45)

        # Initial plot for the unrolled grid
        im = ax2.imshow(deposits, cmap='Purples', aspect='auto')
        ax2.set_title('Unrolled Input', fontsize=18)
        ax2.set_xlabel(r"i$\eta$", fontsize=18)
        ax2.set_ylabel(r"i$\phi$", fontsize=18)

        # Draw red square in unrolled image
        _draw_red_square(ax2, 12, 6)

        cbar = ax2.figure.colorbar(im, ax=ax2)
        cbar.set_ticks([])
        cbar.ax.set_ylabel(r"Energy deposit", fontsize=18)
        plt.xticks([])
        plt.yticks([])

        if not make_animation:
            self._save_fig(name)
            return

        rate_3d = 4
        n_frames = n_phi * rate_3d

        def update(frame):
            # Update the angle of the cylinder
            angle = (-1) * frame * (2 * np.pi / n_frames)  # Total 100 frames for one full rotation
            new_theta = theta + angle  # Shift angles for the cylinder surface

            # new_Theta, _ = np.meshgrid(new_theta, z_cyl)
            _, new_Theta = np.meshgrid(z_cyl, new_theta)

            # Update the cylinder surface with new theta
            X_cyl_new = np.cos(new_Theta)  # New X-coordinates
            Y_cyl_new = np.sin(new_Theta)  # New Y-coordinates

            ax1.cla()  # Clear the previous plot
            ax1.plot_surface(Z_cyl, X_cyl_new, Y_cyl_new, facecolors=plt.cm.Purples(deposits), 
                             rstride=1, cstride=1, alpha=0.6, linewidth=0)
            ax1.set_axis_off() 
            ax1.set_title('CMS Calorimeter (Schematic)', fontsize=18)
            _draw_angle_arcs(ax1, phi_=np.rad2deg(angle)+45)

            # Update the unrolled image based on the rolling motion
            if (frame % rate_3d) == 0:
                shift = int(frame / rate_3d) % n_phi  # Shift amount for unrolled grid
                shifted_Z = np.roll(deposits, shift, axis=0)  # Apply circular permutation along y-axis
                im.set_array(shifted_Z)  # Update the image

                _draw_red_square(ax2, 12, (6+shift) % n_phi)

        # Create the animation
        ani = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=100)

        if self.interactive:
            plt.show()
        else:
            ani.save(
                f"{self.output_dir}/{self._parse_name(name)}.gif"
            )

        plt.close()


    def plot_rate_vs_threshold(
        self,
        scores: List[npt.NDArray],
        baseline_scores: List[npt.NDArray],
        labels: List[str],
        category_labels: tuple[str] = ("Before", "After"),
        name: str = "trigger-rate",
        ylabel: str = "Trigger Rate [kHz]",
    ):
        """
        Plot the trigger rate vs threshold for different versions of the model including a ratio plot.
        Args:
            scores: List of arrays, each containing the ZB anomaly scores a model.
            labels: List of labels for each model.
            name: Name for the output figure.
            baseline_scores: list of baseline scores for comparison.
            category_labels: Optional two-tuple of category labels for the second legend.
            ylabel: Optional alternative Y-axis label.
        """
        # Create figure with GridSpec for subplot control
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        
        # Create main plot and ratio plot with shared x-axis
        ax_main = fig.add_subplot(gs[0])
        ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)
                
        # plot main lines        
        for i, (s, b, l, c) in enumerate(zip(scores, baseline_scores, labels, self.model_colors)):
            thresholds, fractions = get_fractions_above_threshold(s)
            baseline_thresholds, baseline_fractions = get_fractions_above_threshold(b)
            
            ax_main.plot(thresholds, fractions * 28610, ls='-', label=l, color=c)
            ax_main.plot(baseline_thresholds, baseline_fractions * 28610, ls='--', color=c)
                        
            # For ratio plot, we need to interpolate to get values at the same thresholds
            # Let's create a common x-axis (ratio_thresholds) for interpolation
            min_threshold = max(min(thresholds), min(baseline_thresholds))
            max_threshold = min(max(thresholds), max(baseline_thresholds))
            ratio_thresholds = np.linspace(min_threshold, max_threshold, 100)
            
            # Interpolate rates for before and after at these thresholds
            before_interp = np.interp(ratio_thresholds, thresholds, fractions)
            after_interp = np.interp(ratio_thresholds, baseline_thresholds, baseline_fractions)
            
            # Calculate and plot ratio
            ratio = before_interp / after_interp
            ax_ratio.plot(ratio_thresholds, ratio, color=c)
                            
        first_legend = ax_main.legend(loc='upper right')
        ax_main.add_artist(first_legend)

        if category_labels is not None:
            category_handles = [
                Line2D([0], [0], color='grey', linestyle='-', lw=2),
                Line2D([0], [0], color='grey', linestyle='--', lw=2)
            ]
            # Add second legend (styles in grey)
            ax_main.legend(category_handles, category_labels, loc='lower left')
        
        # Set log scale for main plot
        ax_main.set_yscale("log")
        ax_main.set_ylabel(ylabel)
        ax_main.grid(True, linestyle='--', alpha=0.7)
        
        # Remove x-ticks from main plot (will be in ratio plot)
        plt.setp(ax_main.get_xticklabels(), visible=False)
        
        # Configure ratio plot
        ax_ratio.axhline(y=1.0, color='black', linestyle='-', alpha=0.5)  # Reference line at ratio = 1
        ax_ratio.set_ylabel("Ratio")
        ax_ratio.set_xlabel("Score Threshold")
        ax_ratio.grid(True, linestyle='--', alpha=0.7)
        
        # Set reasonable y-limits for ratio plot
        ax_ratio.set_yscale("log")
        
        # Add CMS text to main plot
        hep.cms.text('Preliminary', loc=0, ax=ax_main)
        hep.cms.lumitext(r'2024 (13.6 TeV)', ax=ax_main)
        
        plt.tight_layout()
        self._save_fig(name)


    def make_scatter_density_plot(
        self,
        teacher_scores: npt.NDArray,
        student_scores: npt.NDArray,
        name: str = "cicada_vs_teacher_score",
        xlabel: str = "Teacher Score",
        ylabel: str = "Student Score",
        cutoff: float = 150,
    ):
        
        print(f"{teacher_scores.shape=}")
        print(f"{student_scores.shape=}")
        plt.figure(figsize=(9.5, 6))
        plt.hexbin(teacher_scores, student_scores, gridsize=50, cmap='Blues', bins='log', mincnt=1)

        # add digonal line in grey
        plt.plot([0, cutoff], [0, cutoff], color='grey', linestyle='--', linewidth=1, alpha=0.8)
        plt.colorbar()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.gca().set_xlim(0, cutoff)
        plt.gca().set_ylim(0, cutoff)

        hep.cms.text(self.cms_text, loc=0)
        hep.cms.lumitext(self.lumi_text)

        self._save_fig(name)


    def make_scatter_plot(
        self,
        teacher_scores: List[npt.NDArray],
        student_scores: List[npt.NDArray],
        labels: List[str],
        name: str = "cicada_vs_teacher_score",
        xlabel: str = "Teacher Score",
        ylabel: str = "Student Score",
        cutoff: float = 150,
    ):  
        plt.figure(figsize=(9.5, 6))

        for t, s, l in zip(teacher_scores, student_scores, labels):
            plt.scatter(t.flatten(), s.flatten(), alpha=0.1, label=l)

        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(0, cutoff)
        plt.ylim(0, cutoff)

        # add digonal line in grey
        plt.plot([0, cutoff], [0, cutoff], color='grey', linestyle='--', linewidth=1, alpha=0.8)

        hep.cms.text(self.cms_text, loc=0)
        hep.cms.lumitext(self.lumi_text)

        self._save_fig(name)