#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

extern void calcular_correlacion(int ngal1, int ngal2, int nbins, const double min_rp, const double max_rp, const double min_pi, const double max_pi,
                           const double* x1, const double* y1, const double* z1, const double* x2, const double* y2, const double* z2, double* true_npairs) {

    int i, j;
    const double bin_size_rp = (max_rp - min_rp) / nbins;
    const double bin_size_pi = (max_pi - min_pi) / nbins;
    const int NTHREADS = omp_get_max_threads();
    double **tmp_npairs;

    tmp_npairs = (double **) calloc(NTHREADS,sizeof(double *));
    for (i = 0; i < NTHREADS; i++) 
        tmp_npairs[i] = (double *) calloc(nbins*nbins,sizeof(double));
    
    for (j = 0; j < nbins*nbins; j++)
        true_npairs[j] = 0;

    #pragma omp parallel for schedule(dynamic) num_threads(NTHREADS) \
    default(none) private(j) shared(stdout,NTHREADS,\
    ngal1,ngal2,nbins,min_rp,bin_size_rp,min_pi,bin_size_pi,\
    x1,y1,z1,x2,y2,z2,tmp_npairs)
    for (i = 0; i < ngal1; i++) 
    {
       int tid = omp_get_thread_num();

       if(i % 5000 == 0) fprintf(stdout,"%d/%d\n",i,ngal1);

        for (j = 0; j < ngal2; j++) {
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
            rsq -= rpar * rpar;     // Corregir por la componente en rpar
            double rp = sqrt(rsq);  // la separación en r_p

            // Bineado en r_p
            int k_rp = (int)floor((rp - min_rp) / bin_size_rp);
            if (k_rp < 0 || k_rp >= nbins) continue;  // Excluir pares fuera de los límites

            // Bineado en r_parallel (rpar)
            int k_pi = (int)floor((rpar - min_pi) / bin_size_pi);
            if (k_pi < 0 || k_pi >= nbins) continue;  // Excluir pares fuera de los límites

            // Incrementar el número de pares en el bin correspondiente

            tmp_npairs[tid][k_rp * nbins + k_pi] += 1;
        }
    }

    for (i = 0; i < NTHREADS; i++)
    {
        for (j = 0; j < nbins*nbins; j++)
            true_npairs[j] += tmp_npairs[i][j];
        free(tmp_npairs[i]);
    }
    free(tmp_npairs);

}

