#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern void calcular_correlacion(int ngal1, int ngal2, int nbins, double min_sep, double max_sep, double min_pi, double max_pi,
                           double* x1, double* y1, double* z1, double* x2, double* y2, double* z2,
                           double* true_npairs) {

    double log_min_sep = min_sep;//log(min_sep);
    double log_max_sep = max_sep;//log(max_sep);
    double bin_size_rp = (log_max_sep - log_min_sep) / nbins;
    double bin_size_pi = (max_pi - min_pi) / nbins;

    // Inicializar la matriz de pares
    for (int i = 0; i < nbins * nbins; i++) {
        true_npairs[i] = 0;
    }
    for (int i = 0; i < ngal1; i++) {
        for (int j = 0; j < ngal2; j++) {
            // Cálculo de la separación en r_p (en el plano del cielo)
            double dx = x1[i] - x2[j];
            double dy = y1[i] - y2[j];
            double dz = z1[i] - z2[j];
            double sx = x1[i] + x2[j];
            double sy = y1[i] + y2[j];
            double sz = z1[i] + z2[j];

            double rsq = dx * dx + dy * dy + dz * dz;
            double Lsq = sx * sx + sy * sy + sz * sz;
            double rpar = fabs(dx * sx + dy * sy + dz * sz) / sqrt(Lsq);  // Separación en la dirección paralela
            rsq -= rpar * rpar;  // Corregir por la componente en rpar
            double logr = sqrt(rsq);//0.5 * log(rsq);  // Logaritmo de la separación en r_p
            printf("%f %f \n",logr, rpar);
            // Bineado en r_p
            int k_rp = (int)floor((logr - log_min_sep) / bin_size_rp);
            if (k_rp < 0 || k_rp >= nbins) continue;  // Excluir pares fuera de los límites

            // Bineado en r_parallel (rpar)
            int k_pi = (int)floor((rpar - min_pi) / bin_size_pi);
            if (k_pi < 0 || k_pi >= nbins) continue;  // Excluir pares fuera de los límites

            // Incrementar el número de pares en el bin correspondiente
            true_npairs[k_rp * nbins + k_pi] += 1;
        }
    }

}

