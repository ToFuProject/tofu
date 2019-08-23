// C - algorithm for fast sum of a matrix (similar to np.sum(), sum row by row)
#define MAX 8
void sum_rows_blocks(double *orig, double *out, int n_rows, int n_cols){
   int n_blocks=n_rows/MAX;
   int b;
   int j;
   int i;
   for (b=0; b<n_blocks; b++){   //for every block
       double res[MAX]={0.0};        //initialize to 0.0
       for(j=0;j<n_cols;j++){
           for(i=0;i<MAX;i++){   //calculate sum for MAX-rows simultaniously
              res[i]+=orig[(b*MAX+i)*n_cols+j];
           }
       }
       for(i=0;i<MAX;i++){
             out[b*MAX+i]=res[i];
       }
   }  
   //left_overs:
   int left=n_rows-n_blocks*MAX;
   double res[MAX]={0.0};         //initialize to 0.0
   for(j=0;j<n_cols;j++){
       for(i=0;i<left;i++){   //calculate sum for left rows simultaniously
              res[i]+=orig[(n_blocks*MAX)*n_cols+j];
       }
   }
   for(i=0;i<left;i++){
         out[n_blocks*MAX+i]=res[i];
   }
}
