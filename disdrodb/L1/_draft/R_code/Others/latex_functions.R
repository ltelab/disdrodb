require(tables)

printTab = function(tab, caption, ref, fileName=NULL, tab.env="table",
    opts=list(latexleftpad=FALSE, mathmode=FALSE, latexrightpad=TRUE,
        titlerule="\\cmidrule(lr)")) {

    ## Print a 'tabular' table with a caption and reference.

    if(!is.null(fileName))
        sink(file=fileName)

    cat(paste("\\begin{", tab.env, "}[h!]", sep=""))
    cat("\\centering")
    cat("{\\footnotesize")
    latex(tab, options=opts)
    cat(paste("}\n\\caption{", caption, "}", sep=""))
    cat(paste("\n\\label{", ref, "}", sep=""))
    cat(paste("\n\\end{", tab.env, "}", sep=""))

    if(!is.null(fileName))
        sink(NULL)
}
