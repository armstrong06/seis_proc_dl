import matplotlib.pyplot as plt
import numpy as np

def plot_average_convergence_history(df_summary, is_p_wave = True, alternate_title = None):
    tols = np.unique(df_summary['tolerance'].values)
    print(tols)
    n_epochs = len(np.unique(df_summary['epoch']))
    min_epoch = np.min(np.unique(df_summary['epoch']))
    print(min_epoch)
    accuracies = np.zeros([n_epochs, 2])
    recalls = np.zeros([n_epochs, 2])
    precisions = np.zeros([n_epochs, 2])
    epochs = np.zeros(n_epochs, dtype='int')
    epochs = np.arange(min_epoch, min_epoch+n_epochs)
    for epoch in epochs:
        df_work = df_summary[df_summary['epoch'] == epoch]
        recalls[epoch, 0] = np.average(df_work['recall'].values)
        recalls[epoch, 1] = np.std(df_work['recall'].values)
        accuracies[epoch, 0] = np.average(df_work['accuracy'].values)
        accuracies[epoch, 1] = np.std(df_work['accuracy'].values)
        precisions[epoch, 0] = np.average(df_work['precision'].values)
        precisions[epoch, 1] = np.std(df_work['precision'].values)
        
    plt.figure(figsize = (8,6))
    if (not alternate_title is None):
        plt.title(alternate_title)
    else:
        if (is_p_wave):
            plt.title("Average P Convergence History for Diffferent Tolerances")
        else:
            plt.title("Average S Convergence History for Diffferent Tolerances")
    plt.errorbar(epochs+1, recalls[:, 0],    yerr=recalls[:,1],    capsize=3, color='blue',  label='Recall')
    plt.errorbar(epochs+1, precisions[:, 0], yerr=precisions[:,1], capsize=3, color='red',   label='Precision')
    plt.errorbar(epochs+1, accuracies[:, 0], yerr=accuracies[:,1], capsize=3, color='black', label='Accuracy')
    plt.xlim(min_epoch, max(epochs)+2)
    print(epochs)
    plt.legend()
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.show()

def plot_convergence_history(df_test_summary, tol = 0.5, 
                             is_p_wave = True,
                             alternative_title = None):
    df_work = df_test_summary[ np.abs(df_test_summary['tolerance'].values - tol) < 1.e-4 ]
    if (df_work.shape[0] < 1):
        print("Nothing found for tolerance = ", tol)
        
    fig, ax1 = plt.subplots(figsize=(8,6))
    if (alternative_title is None):
        if (is_p_wave):
            ax1.set_title("P Convergence History for Tolerance = %.2f"%tol)
        else:
            ax1.set_title("S Convergence History for Tolerance = %.2f"%tol)
    else:
        ax1.set_title(alternative_title)
    p1, = ax1.plot(df_work['epoch']+1, df_work['recall'], '-x', color='blue', label='Recall')
    p2, = ax1.plot(df_work['epoch']+1, df_work['precision'], '-o', color='red', label='Precision')
    p3, = ax1.plot(df_work['epoch']+1, df_work['accuracy'], '-+', color='black', label='Accuracy')
    ax1.set_ylabel("Testing Value")
    ax1.set_xlim(min(df_work['epoch']), max(df_work['epoch']+2))
    
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    p4, = ax2.plot(df_work['epoch']+1, df_work['training_loss'], linestyle='-.', color='green', label='BCE Loss')
    ax2.set_ylabel("Training BCE Loss")
    lines = [p1,p2,p3,p4]
    ax1.legend(lines, [l.get_label() for l in lines], loc='lower right')
    #ax2.legend()
    
    ax1.grid(True)
    plt.show()

def plot_residuals(df_resid, probability = 0.905, ylim=(-20, 20)):
    #print(df_resid.head(5))
    df_work = df_resid[np.abs(df_resid['tolerance'] - probability) < 1.e-4]
    plt.figure(figsize=(8,6))
    plt.title("P Average Residual in Samples for Tolerance %.2f"%probability)
    plt.errorbar(df_work['epoch'].values + 1, 
                 df_work['residual_mean'].values,
                 yerr = df_work['residual_std'].values, fmt='o',
                 color = 'black', capthick=1, capsize=4, label='Mean of All Residuals')
    plt.errorbar(df_work['epoch'].values + 1, 
                 df_work['trimmed_residual_mean'].values,
                 yerr = df_work['trimmed_residual_std'].values, fmt='x',
                 color = 'red', capthick=1, capsize=4, label='Mean of Residuals in Outer Fence')
    plt.ylim(ylim[0], ylim[1])
    plt.yticks(np.arange(ylim[0], ylim[1]+10, 10))
    plt.xlim(np.min(df_work['epoch'])-1, max(df_work.epoch)+2)
    plt.legend()
    plt.grid(True)
    #print(np.min(df_resid['true_lag']), np.max(df_resid['true_lag']))
    #plt.scatter(np.abs(df_work['residual']), df_work['probability'])
    plt.show()
    print(df_work['residual_mean'].values)

def plot_seperate_convergence_history(df_test_summary, tol = 0.5, 
                             alternative_title = None):

    df_work = df_test_summary[ np.abs(df_test_summary['tolerance'].values - tol) < 1.e-4 ]
    if (df_work.shape[0] < 1):
        print("Nothing found for tolerance = ", tol)
    
    for metric in ["recall", "accuracy", "precision"]:
        fig, ax1 = plt.subplots(figsize=(8,6))
        if (alternative_title is None):
            ax1.set_title("%s for Tolerance = %.2f"%(metric, tol))
        else:
            ax1.set_title(alternative_title)


        p1, = ax1.plot(df_work['epoch']+1, df_work[metric], '-x', label='combined')
        lines=[p1]
        if metric != "precision":
            p2, = ax1.plot(df_work['epoch']+1, df_work[f'ceq_{metric}'], '-x', label='ceq')
            p3, = ax1.plot(df_work['epoch']+1, df_work[f'heq_{metric}'], '-x', label='heq')
            p4, = ax1.plot(df_work['epoch']+1, df_work[f'cbl_{metric}'], '-x', label='cbl')

            lines = [p1,p2,p3,p4]
            if metric == "accuracy":
                p5, = ax1.plot(df_work['epoch']+1, df_work[f'noise_{metric}'], '-x', label='noise')
                lines.append(p5)


        ax1.set_ylabel("Testing Value")
        ax1.set_xlim(np.min(df_work["epoch"]), max(df_work['epoch']+1))

        ax1.legend(lines, [l.get_label() for l in lines], loc='lower right')
        #ax2.legend()
        #ax1.set_ylim([0.90, 1.01])
        ax1.grid(True)
        plt.show()

def plot_residual_hist(df_valid_resid, opt_model, meta_df=None, bins=np.arange(-50, 50, 1), 
                        epoch_key="epoch"):
    plt.figure(figsize=(8,6))
    df_epoch = df_valid_resid[df_valid_resid['model'] == opt_model]

    plt.hist(df_epoch[(df_epoch["true_lag"] > -1)].residual, bins=bins, edgecolor='black')
    if meta_df is not None:
        current_eq_rows = np.arange(0, len(meta_df))[ (meta_df['event_type'] == 'le') & (meta_df['evid'] >= 60000000)]
        current_blast_rows = np.arange(0, len(meta_df))[meta_df['event_type'] == 'qb']
        historical_eq_rows = np.arange(0, len(meta_df))[(meta_df['event_type'] == 'le') & (meta_df['evid'] < 60000000)]
        plt.hist(df_epoch.iloc[current_eq_rows].residual, bins=bins, edgecolor='black', label="ceq")
        plt.hist(df_epoch.iloc[historical_eq_rows].residual, bins=bins, edgecolor='black', label="heq")
        plt.hist(df_epoch.iloc[current_blast_rows].residual, bins=bins, edgecolor='black', label="cqb")

    plt.title("Pick Residual Distribution for Epoch %d"%(opt_model+1))

    plt.xlabel("Pick Residual Samples")
    plt.ylabel("Counts")
    plt.xlim(-50, 50)
    plt.show()

def plot_mew_hist(df_valid_resid, epoch, bins=np.arange(-50, 50, 1), epoch_key="epoch"):
    plt.figure(figsize=(8,6))
    df_epoch = df_valid_resid[df_valid_resid[epoch_key] == epoch]

    even_rows = np.arange(0, len(df_epoch), 2)
    odd_rows = np.arange(1, len(df_epoch), 2)

    plt.hist(df_epoch.iloc[even_rows].residual, bins=bins, edgecolor='black', label="first_event")
    plt.hist(df_epoch.iloc[odd_rows].residual, bins=bins, edgecolor='black', alpha=0.5, label="second_event")

    plt.title("Pick Residual Distribution for Epoch %d"%(epoch+1))
    plt.xlabel("Pick Residual Samples")
    plt.ylabel("Counts")
    #plt.xlim(-50, 50)
    plt.legend()
    plt.show()

def plot_magnitude_difference_vs_residual(df_valid_resid, df_valid, epoch, epoch_key="epoch"):
    df_epoch = df_valid_resid[df_valid_resid[epoch_key] == epoch]
    
    even_rows = np.arange(0, len(df_epoch), 2)
    odd_rows = np.arange(1, len(df_epoch), 2)

    magnitude_diffs = (df_valid.iloc[even_rows].magnitude.values - df_valid.iloc[odd_rows].magnitude.values)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].scatter(magnitude_diffs, df_epoch.iloc[even_rows].residual, alpha=0.2)
    axes[1].scatter(magnitude_diffs, df_epoch.iloc[odd_rows].residual, alpha=0.2)

    axes[0].set_ylabel("Pick Residual (samples)")
    axes[1].set_xlabel("Magnitude Difference (First event - second event)")
    axes[1].xaxis.set_label_coords(-0.15, -.1)
    axes[0].set_ylim([-520, 520])
    axes[1].set_ylim([-520, 520])
    axes[1].set_yticks([])
    axes[0].set_title("First Event")
    axes[1].set_title("Second Event")

    plt.figure()
    plt.scatter(magnitude_diffs, df_epoch.iloc[odd_rows].residual, alpha=0.2, label="Second event")
    plt.scatter(magnitude_diffs, df_epoch.iloc[even_rows].residual, alpha=0.2, label="First event")

    plt.ylabel("Pick Residual (samples)")
    plt.xlabel("Mangitude Difference (First event - second event)")
    plt.legend()

def plot_precision_recall(df, opt_model, is_p_wave = True):
    df_work = df[df['epoch'] == opt_model]
    precision = df_work['precision'].values
    recall = df_work['recall'].values
    tol = df_work['tolerance'].values
    #if ()
    if (is_p_wave):
        plt.scatter(recall, precision, c=tol, marker='o', label='P')
    else:
        plt.scatter(recall, precision, c=tol, marker='^', label='S')
    plt.title("Precision-Recall for Epoch %d"%(opt_model+1))  
    cbar = plt.colorbar()
    cbar.set_label("Posterior Probability")
#     plt.xlim(0.8, 1)
#     plt.ylim(0.97, 1.01)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()
    #print(df_work)

def plt_snr_probs(df_resid, opt_model):
    df_resid_om = make_opt_model_df(df_resid, opt_model)

    snrs = df_resid_om['snr'].values
    probas = df_resid_om['probability'].values
    perm = np.argsort(probas)
    snrs = snrs[perm[:]]
    probas = probas[perm[:]]
    #proba_cum = np.cumsum(probas)/np.sum(probas)
    xvals = np.linspace(min(probas), max(probas), 600)
    pcts = np.zeros(len(xvals))
    for i in range(len(xvals)):
        is_true = (probas <= xvals[i])*1
        pcts[i] = np.sum(is_true)/len(probas)

    fig, ax1 = plt.subplots(figsize=(8,6))
    ax1.scatter(probas, snrs) 
    ax1.set_title("P Pick SNR to Posterior Probability")
    ax1.set_xlabel('Posterior Probability')
    ax1.set_ylabel('SNR')
    ax1.set_xlim(-0.05,1.05)

    ax2 = ax1.twinx()
    ax2.plot(xvals, pcts, color='black')
    ax2.set_ylabel("Cumulative Fraction of Picks")
    ax2.set_ylim(0,1)

    plt.show()


def plot_snr_residuals(df_resid, opt_model):
    df_resid_om = make_opt_model_df(df_resid, opt_model)
    snrs = df_resid_om['snr'].values
    residuals = df_resid_om['residual'].values

    perm = np.argsort(residuals)
    snrs = snrs[perm[:]]
    residuals = residuals[perm[:]]

    xvals = np.linspace(min(residuals), max(residuals), 600)
    pcts = np.zeros(len(xvals))
    for i in range(len(xvals)):
        is_true = (residuals <= xvals[i])*1
        pcts[i] = np.sum(is_true)/len(residuals)
    #resid_cum = np.cumsum(residuals)/np.sum(residuals)

    #plt.scatter(df_test_resid_om['residual'], df_test_resid_om['snr'])
    #plt.scatter(residuals, snrs)
    #plt.title("P Pick SNR vs. Residual")
    #plt.xlabel('Residual (Samples)')
    #plt.ylabel('SNR')
    #plt.xlim(-100,100)
    #plt.show()

    fig, ax1 = plt.subplots(figsize=(8,6))
    ax1.scatter(residuals, snrs) 
    ax1.set_title("P Pick SNR to Residual")
    ax1.set_xlabel('Residual (Samples)')
    ax1.set_ylabel('SNR')
    ax1.set_xlim(-100,100)

    ax2 = ax1.twinx()
    ax2.plot(xvals, pcts, color='black')
    ax2.set_ylabel("Cumulative Fraction of Picks")
    ax2.set_ylim(0,1)
    plt.show()

def make_opt_model_df(df_resid, opt_model):
    df_resid_om = df_resid[df_resid['epoch'] == opt_model]
    df_resid_om = df_resid_om[ ['residual', 'probability', 'snr'] ]
    df_resid_om['absolute_residual'] = np.abs(df_resid_om['residual'].values)
    return df_resid_om