#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

extern void calcular_correlacion(int ngal1, int ngal2, int nbins, const double min_rp, const double max_rp, const double min_pi, const double max_pi,
                           const double* ra1, const double* dec1, const double *r1, const double* ra2, const double* dec2, const double *r2, double* true_npairs) {

    int i, j;
    const double bin_size_rp = (max_rp - min_rp) / nbins;
    const double bin_size_pi = (max_pi - min_pi) / nbins;
    const int NTHREADS = omp_get_max_threads();
    double **tmp_npairs;
    double *tmp_sin, *tmp_cos, *tmp_rsq;

    fprintf(stdout,"rp parameters %.2f %.2f %.2f\n",min_rp,max_rp,bin_size_rp);
    fprintf(stdout,"pi parameters %.2f %.2f %.2f\n",min_pi,max_pi,bin_size_pi);
    
    tmp_npairs = (double **) calloc(NTHREADS,sizeof(double *));
    for (i = 0; i < NTHREADS; i++) 
        tmp_npairs[i] = (double *) calloc(nbins*nbins,sizeof(double));
    
    for (j = 0; j < nbins*nbins; j++)
        true_npairs[j] = 0;

    tmp_sin = (double *) calloc(ngal2,sizeof(double));
    tmp_cos = (double *) calloc(ngal2,sizeof(double));
    tmp_rsq = (double *) calloc(ngal2,sizeof(double));

    #pragma omp parallel for schedule(dynamic) num_threads(NTHREADS) \
    default(none) shared(stdout,ngal2,ra2,dec2,r2,tmp_sin,tmp_cos,tmp_rsq)
    for (j = 0; j < ngal2; j++) 
    {
        tmp_sin[j] = sin(dec2[j]);
        tmp_cos[j] = cos(dec2[j]);
        tmp_rsq[j] = r2[j]*r2[j];
    } 

    #pragma omp parallel for schedule(dynamic) num_threads(NTHREADS) \
    default(none) private(j) shared(stdout,NTHREADS,\
    ngal1,ngal2,nbins,min_rp,bin_size_rp,min_pi,bin_size_pi,\
    ra1,dec1,r1,ra2,dec2,r2,tmp_sin,tmp_cos,tmp_rsq,tmp_npairs)
    for (i = 0; i < ngal1; i++) 
    {
       int tid = omp_get_thread_num();

       if(i % 5000 == 0) fprintf(stdout,"%d/%d\n",i,ngal1);
       
       double ri    = r1[i];
       double rsq_i = r1[i]*r1[i];
       double sin_i = sin(dec1[i]);
       double cos_i = cos(dec1[i]);
       
       for (j = 0; j < ngal2; j++) 
       {
            double mu = sin_i * tmp_sin[j] + cos_i * tmp_cos[j] * cos((ra1[i]-ra2[j]));
            double ang = 0.5*acos(mu);
            double pi  = fabs((ri-r2[j])*cos(ang));
            double rp  = fabs((ri+r2[j])*sin(ang)); 
 
            // Bineado en r_p
            int k_rp = (int)floor((rp - min_rp) / bin_size_rp);
            if (k_rp < 0 || k_rp >= nbins) continue;  // Excluir pares fuera de los límites

            // Bineado en r_parallel
            int k_pi = (int)floor((pi - min_pi) / bin_size_pi);
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
    free(tmp_sin);
    free(tmp_cos);
    free(tmp_rsq);
}

extern void calcular_autocorrelacion(int ngal1, int nbins, const double min_rp, const double max_rp, const double min_pi, const double max_pi,
                                     const double* ra1, const double* dec1, const double *r1, double* true_npairs) {

    int i, j;
    const double bin_size_rp = (max_rp - min_rp) / nbins;
    const double bin_size_pi = (max_pi - min_pi) / nbins;
    const int NTHREADS = omp_get_max_threads();
    double **tmp_npairs;
    double *tmp_sin, *tmp_cos, *tmp_rsq;

    fprintf(stdout,"rp parameters %.2f %.2f %.2f\n",min_rp,max_rp,bin_size_rp);
    fprintf(stdout,"pi parameters %.2f %.2f %.2f\n",min_pi,max_pi,bin_size_pi);

    tmp_npairs = (double **) calloc(NTHREADS,sizeof(double *));
    for (i = 0; i < NTHREADS; i++) 
        tmp_npairs[i] = (double *) calloc(nbins*nbins,sizeof(double));
    
    for (j = 0; j < nbins*nbins; j++)
        true_npairs[j] = 0;
    
    tmp_sin = (double *) calloc(ngal1,sizeof(double));
    tmp_cos = (double *) calloc(ngal1,sizeof(double));
    tmp_rsq = (double *) calloc(ngal1,sizeof(double));

    #pragma omp parallel for schedule(dynamic) num_threads(NTHREADS) \
    default(none) shared(stdout,ngal1,ra1,dec1,r1,tmp_sin,tmp_cos,tmp_rsq)
    for (j = 0; j < ngal1; j++) 
    {
        tmp_sin[j] = sin(dec1[j]);
        tmp_cos[j] = cos(dec1[j]);
        tmp_rsq[j] = r1[j]*r1[j];
    } 

    #pragma omp parallel for schedule(dynamic) num_threads(NTHREADS) \
    default(none) private(j) shared(stdout,NTHREADS,\
    ngal1,nbins,min_rp,bin_size_rp,min_pi,bin_size_pi,\
    ra1,dec1,r1,tmp_npairs,tmp_sin,tmp_cos,tmp_rsq)
    for (i = 0; i < ngal1-1; i++) 
    {
       int tid = omp_get_thread_num();

       if(i % 5000 == 0) fprintf(stdout,"%d/%d\n",i,ngal1);

       double ri    = r1[i];
       double rsq_i = tmp_rsq[i];
       double sin_i = tmp_sin[i];
       double cos_i = tmp_cos[i];
       
       for (j = i+1; j < ngal1; j++) 
       {            
          //double mu  = sin_i * tmp_sin[j] + cos_i * tmp_cos[j] * cos((ra1[i]-ra1[j]));
          //double r   = sqrt(rsq_i + tmp_rsq[j] - 2*r1[j]*ri*mu);
          //double pi = r*fabs(mu);
          //double rp = r*sqrt(fabs(1 - mu*mu)); 
          
          double mu  = sin_i * tmp_sin[j] + cos_i * tmp_cos[j] * cos((ra1[i]-ra1[j]));
          double ang = 0.5*acos(mu);
          double pi  = fabs((ri-r1[j])*cos(ang));
          double rp  = fabs((ri+r1[j])*sin(ang)); 
 
          //fprintf(stdout,"%.6f %.6f %.6f %.6f\n",mu,180*ang/M_PI,rp,pi);

          // Bineado en r_p
          int k_rp = (int)floor((rp - min_rp) / bin_size_rp);
          if (k_rp < 0 || k_rp >= nbins) continue;  // Excluir pares fuera de los límites

          // Bineado en r_parallel (pi)
          int k_pi = (int)floor((pi - min_pi) / bin_size_pi);
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
    free(tmp_sin);
    free(tmp_cos);
    free(tmp_rsq);
}
