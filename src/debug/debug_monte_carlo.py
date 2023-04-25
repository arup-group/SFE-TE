""" Debug module used for testing Monte Carlo"""

from src import monte_carlo as mc
import matplotlib.pyplot as plt
import seaborn as sns

def simple_plot(inputs):
    sns.set()
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.ravel()
    for i, input in enumerate(inputs['values']):
        axs[i].hist(inputs['values'][input], bins=25, density=True, align='mid', label='sampled data',
                               alpha=0.7)
        axs[i].plot(inputs['curves'][input][:, 0], inputs['curves'][input][:, 1],
                    linestyle='dashed', color='#FF7F0E', label='analytical target')
        axs[i].set_xlabel(input)
    handles, labels = axs[0].get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', ncol=2)
    plt.show()


# Inputs
inputs = {'spr_rate1': ['rect_pulse', 0, 10, 0, 0.33, 0.2, None, None],
          'spr_rate2': ['rect_pulse', 0, 10, 0.33, 0.66, 0.33, None, None],
          'spr_rate3': ['rect_pulse', 0, 10, 0.66, 1, 0.2, None, None]}

rnd = mc.ProbControl(seed=None)
asd = rnd.sample_inputs(inputs=inputs, sample_size=2000)
simple_plot(asd)

print('asd')




