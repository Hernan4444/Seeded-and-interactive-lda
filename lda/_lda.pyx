#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

cdef extern from "gamma.h":
    cdef double lda_lgamma(double x) nogil


cdef double lgamma(double x) nogil:
    if x <= 0:
        with gil:
            raise ValueError("x must be strictly positive")
    return lda_lgamma(x)


cdef int searchsorted(double* arr, int length, double value) nogil:
    """Bisection search (c.f. numpy.searchsorted)

    Find the index into sorted array `arr` of length `length` such that, if
    `value` were inserted before the index, the order of `arr` would be
    preserved.
    """
    cdef int imin, imax, imid
    imin = 0
    imax = length
    while imin < imax:
        imid = imin + ((imax - imin) >> 1)
        if value > arr[imid]:
            imin = imid + 1
        else:
            imax = imid
    return imin


def _sample_topics(int[:] WS, int[:] DS, int[:] ZS, int[:, :] nzw, int[:, :] ndz, int[:] nz,
                   double[:] alpha, double[:] eta, double[:] rands):
    """
    WS: lista con cada palabra entre todos los documentos
    DS: lista con el documento asociado a cada palabra
    ZS: lista con el tópico asociaddo a cada palabra
    nzw: Matriz de tópicos (filas) por palabras (columna)
    ndz: Matriz de documento por tópico
    nz: Lista, cantidad de palabras asignadas al tópico

    alpha y eta lista de hiperparametros para distribución
    rands: lista de números random (entre 0 y 1) para generar valores aleatorios.
    Ej: (2 tópicos)
        Doc0: "a b"
        Doc1: "c d"
        Doc2: "a d"
        -------------
        WS: [a, b, c, d, a, d]
        DS: [0, 0, 1, 1, 2, 2]
        ZS: [0, 1, 1, 0, 1, 0] Tópicos puestos al azar
        
               a b c d
        nzw: [ 1 0 0 2   topico 0
               1 1 1 0 ] topico 1

              T0  T1
        ndz: [ 1   1    Doc0
               1   1    Doc1
               1   1 ]  Doc2

             T0  T1
        nz: [ 3   3]
    
    """
    cdef int i, k, w, d, z, z_new
    cdef double r, dist_cum
    cdef int N = WS.shape[0]
    cdef int n_rand = rands.shape[0]
    cdef int n_topics = nz.shape[0]
    cdef double eta_sum = 0
    cdef double* dist_sum = <double*> malloc(n_topics * sizeof(double))
    if dist_sum is NULL:
        raise MemoryError("Could not allocate memory during sampling.")
    with nogil:
        for i in range(eta.shape[0]):
            eta_sum += eta[i]

        for i in range(N):
            w = WS[i] # Palabra
            d = DS[i] # Documento
            z = ZS[i] # Tópico

            dec(nzw[z, w]) # Desasignar dicha palabra al tópico
            dec(ndz[d, z]) # Desasignar dicho tópico al documento
            dec(nz[z]) # Disminuir palabras totales del tópico

            dist_cum = 0
            for k in range(n_topics):
                """
                Para calcular las probabilidades, se obtiene la probabilidad
                por tópico, pero este se hace de forma acumulada. Es decir,
                Si son 3 tópicos y las probabilidades son: [0.2, 0.5, 0.3]
                La lista final es: [0.2, 0.7, 1]
                De este modo, generamos solo 1 número random entre 0 y 1 (r)
                y buscamos el índice del número más grande y cercano al generado
                Si r <= 0.2, el tópico es 0 (hay 0.2 de probabilidad)
                Si 0.2 < r <= 0.7, el tópico es 1 (hay 0.7 - 0.2 = 0.5 de probabilidad)
                si 0.7 < r, el tópico es 2 (hay 1 - 0.7 = 0.3 de probabilidad)
                """
                # eta is a double so cdivision yields a double
                dist_cum += (ndz[d, k] + alpha[k]) * (nzw[k, w] + eta[w]) / (nz[k] + eta_sum)
                # Falta la división (nzw[k,*] + K*alpha[k]), pero es constante
                # Se puede omitir y no afecta.

                dist_sum[k] = dist_cum
            """
            Puede que el último valor de la lista no sea 1.
            Por lo tanto se adapta el random para que sea entre 0 y dist_sum[-1].
            """
            r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]
            z_new = searchsorted(dist_sum, n_topics, r) # Buscar tópico

            ZS[i] = z_new  # Asignar nuevo tópico
            inc(nzw[z_new, w]) # Aumentar la palabra del tópico
            inc(ndz[d, z_new]) # Aumentar el tópico del documento
            inc(nz[z_new]) # aumentar cantidad de palabras asociadas al tópico

        free(dist_sum)

def _sample_topics_interactive(int[:] WS, int[:] DS, int[:] ZS, int[:, :] nzw, int[:, :] ndz, int[:] nz,
                   double[:] alpha, double[:] eta, double[:] rands, int[:] SW, int[:, :] nzs, double[:, :] nu):
    """
    WS: lista con cada palabra entre todos los documentos
    DS: lista con el documento asociado a cada palabra
    ZS: lista con el tópico asociaddo a cada palabra
    SW: lista con el seed asociaddo a cada palabra, -1 si no hay seed
    nzs: Matriz de tópicos (filas) por seed (columna)
    nzw: Matriz de tópicos (filas) por palabras (columna)
    ndz: Matriz de documento por tópico
    nz: Lista, cantidad de palabras asignadas al tópico

    alpha, eta, nu lista de hiperparametros para distribución
    nu es una lista de tuplas, cada tupla es de la forma (nu, len(seed[s]))
    rands: lista de números random (entre 0 y 1) para generar valores aleatorios.
    
    """
    cdef int i, k, w, d, z, z_new, s
    cdef double r, dist_cum
    cdef int N = WS.shape[0]
    cdef int n_rand = rands.shape[0]
    cdef int n_topics = nz.shape[0]
    cdef double eta_sum = 0
    cdef double nu_sum = 0
    #cdef float aux;
    cdef double* dist_sum = <double*> malloc(n_topics * sizeof(double))
    if dist_sum is NULL:
        raise MemoryError("Could not allocate memory during sampling.")
    with nogil:
        for i in range(eta.shape[0]):
            eta_sum += eta[i]

        for i in range(N):
            w = WS[i] # Palabra
            d = DS[i] # Documento
            z = ZS[i] # Tópico
            s = SW[i] # Seed
            dec(nzw[z, w]) # Desasignar dicha palabra al tópico
            dec(ndz[d, z]) # Desasignar dicho tópico al documento
            dec(nz[z]) # Disminuir palabras totales del tópico

            dist_cum = 0
            
            if (s >= 0):
                nu_sum = nu[s, 0] * nu[s, 1]
                dec(nzs[z, s]) # Desasignar seed al tópico. 

                for k in range(n_topics):
                    dist_cum += (ndz[d, k] + alpha[k]) * (nzs[k, s] + nu[s, 1]*eta[w]) / (nz[k] + eta_sum) *  (nzw[k, w] + nu[s, 0]) / (nzs[k, s] + nu_sum)
                    #aux = (nzw[k, w]) / (nzs[k, s] );
                    #printf("%d %d -> %f -", nzw[k, w], nzs[k, s], aux);
                    dist_sum[k] = dist_cum
                #printf("\n")
            else:
                for k in range(n_topics):
                    dist_cum += (ndz[d, k] + alpha[k]) * (nzw[k, w] + eta[w]) / (nz[k] + eta_sum) 
                    dist_sum[k] = dist_cum

            r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]
            z_new = searchsorted(dist_sum, n_topics, r) # Buscar tópico

            ZS[i] = z_new  # Asignar nuevo tópico
            inc(nzw[z_new, w]) # Aumentar la palabra del tópico
            inc(ndz[d, z_new]) # Aumentar el tópico del documento
            inc(nz[z_new]) # Aumentar cantidad de palabras asociadas al tópico
            if (s >= 0):
                inc(nzs[z_new, s]) # Aumentar el seed al tópico.

        free(dist_sum)


cpdef double _loglikelihood(int[:, :] nzw, int[:, :] ndz, int[:] nz, int[:] nd, double alpha, double eta) nogil:
    cdef int k, d
    cdef int D = ndz.shape[0]
    cdef int n_topics = ndz.shape[1]
    cdef int vocab_size = nzw.shape[1]

    cdef double ll = 0

    # calculate log p(w|z)
    cdef double lgamma_eta, lgamma_alpha
    with nogil:
        lgamma_eta = lgamma(eta)
        lgamma_alpha = lgamma(alpha)

        ll += n_topics * lgamma(eta * vocab_size)
        for k in range(n_topics):
            ll -= lgamma(eta * vocab_size + nz[k])
            for w in range(vocab_size):
                # if nzw[k, w] == 0 addition and subtraction cancel out
                if nzw[k, w] > 0:
                    ll += lgamma(eta + nzw[k, w]) - lgamma_eta

        # calculate log p(z)
        for d in range(D):
            ll += (lgamma(alpha * n_topics) -
                    lgamma(alpha * n_topics + nd[d]))
            for k in range(n_topics):
                if ndz[d, k] > 0:
                    ll += lgamma(alpha + ndz[d, k]) - lgamma_alpha
        return ll
