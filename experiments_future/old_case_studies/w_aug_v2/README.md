**Hab ich ihm am `11.06` geschickt**

Originaler Name: `w_aug_v2`



Ich hab ein environment.yaml beigelegt. Ich hab die letzten paar Tage noch damit rumgespielt und bin mir 95% sicher, dass zumindest das Jax Module passt (siehe rbm_pha_grad_spike).

Das Problem liegt also ziemlich sicher am Training. Die Variante mit der Fisher Matrix spukt einen loss aus, der sich garnirgens hinbewegt. Am vielversprechendsten ist aber vanilla adamw mit einer alternativen freien Energie finde ich (siehe rbm_pha_fourier).

Zum ShadowRBM. Da habe ich leider das Problem, dass ich die persistent chains rausgenommen habe und die waren recht wichtig. Das Problem ist, dass ich die Persistent Chain nicht in den nächsten Batch mitnehmen kann, da da ja alle Samples eine andere Basis haben.



