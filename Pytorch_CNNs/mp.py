import multiprocessing as mp
from multiprocessing import Pool, Manager


def worker_func(data, filename, lock):
    datax= data
    with lock:
        with open(filename,'a') as fn:
            fn.write(f"write line example {datax}\n")


def main():
    workers = 32    
    with Manager() as manager:        
        lock = manager.Lock() 
        pool_args=[(item, "file.csv", lock) for item in range(0,1000)]       
        with Pool(processes=workers) as pool:            
            results = pool.starmap_async(worker_func, pool_args) 
            results.wait()

if __name__=="__main__":
    main()