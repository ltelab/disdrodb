## Functions to deal with CSV files.
## Tim Raupach <tim.raupach@epfl.ch>

require(data.table)

readCSVStack = function(dir, header=TRUE, as.is=TRUE, ...) {
    ## Read data from all files in a directory and return a data.table
    ## containing all data.
    ##
    ## Args:
    ##  dir: The directory to read from.
    ##  ...: Extra arguments to read.table().
    ##
    ## Returns: a data.table containing all data from all files.

    files = list.files(dir, full.names=TRUE)

    dat = NULL
    for(f in files) {
        d = data.table(read.table(f, header=header, as.is=as.is, ...))
        dat = rbind(dat, d)
    }
    
    return(dat)
}

readCSVStack_bySubdir = function(dir, ...) {
    ## Read data separately from sub directories.
    ##
    ## Args:
    ##   dir: The parent directory.
    ##   ...: Extra arguments to readCSVStack().
    ##
    ## Returns: a data.table containing all data from all files, with
    ## column "subdir" for each subdirectory.

    dirs = list.files(dir, full.names=TRUE)
    dat = NULL
    for(subdir in dirs) {
        dat = rbind(dat, data.table(readCSVStack(dir=subdir, ...), subdir=basename(subdir)))
    }

    return(dat)
}
