import os
import optuna

# optunaの出力をテキストファイルに保存

def optuna_log_write(study,n_trials,file_name="log",log_dir="../logs"):
    tuning_time = (study.trials[n_trials-1].datetime_complete-study.trials[0].datetime_start).total_seconds()
    try:
        with open(f"{log_dir}/{file_name}/{file_name}.txt","w") as file:
            for i in range(n_trials):
                file.write("Trial  :"+str(study.trials[i].number)+"\n")
                file.write("Value  :"+str(study.trials[i].values[0])+"\n")
                file.write("Params :"+str(study.trials[i].params)+"\n\n")

            file.write(f"Best trial is {study.best_trial.number}\n")
            file.write(f"Value :{study.best_value}\n")
            file.write(f"Params:{study.best_params}\n\n")
            file.write(f"Total tuning time is {tuning_time}s")
            

        fig_history = optuna.visualization.plot_optimization_history(study)
        fig_history.write_image(f"{log_dir}/{file_name}/history.pdf")
        fig_importance = optuna.visualization.plot_param_importances(study)
        fig_importance.write_image(f"{log_dir}/{file_name}/importance.pdf")
      
    except FileNotFoundError as e:
        print(f"Make directory. name '{log_dir}/{file_name}'")