require(motifmatchr)
require(GenomicRanges)
require(BSgenome.Dmelanogaster.UCSC.dm3)
t1 <- readRDS("CP_motifs_PWM.rds")
df = read.table("coordinates_p65.txt",sep=",",header=TRUE,colClasses = c("character", "numeric", "numeric", "character"))
df$start <- df$start+1
seqinfo1= Dmelanogaster@seqinfo
Sequences <- with(df, GRanges(seqnames = Rle(chr), ranges = IRanges(start, end), strand = strand,seq=seq),seqinfo=seqinfo1)
Sequences$ID <- paste(Sequences@seqnames, Sequences@ranges@start, Sequences@ranges@start+Sequences@ranges@width-1,sep="_")
motif_ix <- matchMotifs(t1$Pwms_perc,
                        Sequences$seq,
                         genome = "BSgenome.Dmelanogaster.UCSC.dm3", p.cutoff = 1e-4, bg="genome", out = "positions")
print(motif_ix)
motif_ix_pos2 <- lapply(motif_ix, function(motif_xy){
names(motif_xy) <- Sequences$ID
motif_xy <- motif_xy[sapply(motif_xy, length)>0] # remove sequences without motif
  motif_xy1 <- GRanges(names(unlist(motif_xy)), #Use sequence ID as seqnames
            IRanges(start(unlist(motif_xy)), end(unlist(motif_xy))),
            strand = mcols(unlist(motif_xy))$strand,
            score = mcols(unlist(motif_xy))$score)
  return(motif_xy1)
 })
names(motif_ix_pos2) <- c("Ohler1",
"DRE",
"TATAbox",
"INR",
"Ebox",
"Ohler6",
"Ohler7",
"Ohler8",
"DPE",
"MTE",
"TCT",
"DPE_extended",
"DPE_Kadonaga",
"BREu",
"BREd",
"DCE1",
"DCE2",
"DCE3",
"TC_17_Zabidi")
for (i in 1:length(motif_ix_pos2)) {
write.table( x = data.frame(motif_ix_pos2[[i]]), file = paste0("./",names(motif_ix_pos2)[i], "_p65.csv"), sep=",", col.names=TRUE, row.names=FALSE, quote=FALSE )
}
print("done csv")
df = read.table("coordinates_p300.txt",sep=",",header=TRUE,colClasses = c("character", "numeric", "numeric", "character"))
df$start <- df$start+1
seqinfo1= Dmelanogaster@seqinfo
Sequences <- with(df, GRanges(seqnames = Rle(chr), ranges = IRanges(start, end), strand = strand,seq=seq),seqinfo=seqinfo1)
Sequences$ID <- paste(Sequences@seqnames, Sequences@ranges@start, Sequences@ranges@start+Sequences@ranges@width-1,sep="_")
motif_ix <- matchMotifs(t1$Pwms_perc,
                        Sequences$seq,
                         genome = "BSgenome.Dmelanogaster.UCSC.dm3", p.cutoff = 1e-4, bg="genome", out = "positions")
print(motif_ix)
motif_ix_pos2 <- lapply(motif_ix, function(motif_xy){
names(motif_xy) <- Sequences$ID
motif_xy <- motif_xy[sapply(motif_xy, length)>0] # remove sequences without motif
  motif_xy1 <- GRanges(names(unlist(motif_xy)), #Use sequence ID as seqnames
            IRanges(start(unlist(motif_xy)), end(unlist(motif_xy))),
            strand = mcols(unlist(motif_xy))$strand,
            score = mcols(unlist(motif_xy))$score)
  return(motif_xy1)
 })
names(motif_ix_pos2) <- c("Ohler1",
"DRE",
"TATAbox",
"INR",
"Ebox",
"Ohler6",
"Ohler7",
"Ohler8",
"DPE",
"MTE",
"TCT",
"DPE_extended",
"DPE_Kadonaga",
"BREu",
"BREd",
"DCE1",
"DCE2",
"DCE3",
"TC_17_Zabidi")
for (i in 1:length(motif_ix_pos2)) {
write.table( x = data.frame(motif_ix_pos2[[i]]), file = paste0("./",names(motif_ix_pos2)[i], "_p300.csv"), sep=",", col.names=TRUE, row.names=FALSE, quote=FALSE )
}
print("done csv")
df = read.table("coordinates_gfzf.txt",sep=",",header=TRUE,colClasses = c("character", "numeric", "numeric", "character"))
df$start <- df$start+1
seqinfo1= Dmelanogaster@seqinfo
Sequences <- with(df, GRanges(seqnames = Rle(chr), ranges = IRanges(start, end), strand = strand,seq=seq),seqinfo=seqinfo1)
Sequences$ID <- paste(Sequences@seqnames, Sequences@ranges@start, Sequences@ranges@start+Sequences@ranges@width-1,sep="_")
motif_ix <- matchMotifs(t1$Pwms_perc,
                        Sequences$seq,
                         genome = "BSgenome.Dmelanogaster.UCSC.dm3", p.cutoff = 1e-4, bg="genome", out = "positions")
print(motif_ix)
motif_ix_pos2 <- lapply(motif_ix, function(motif_xy){
names(motif_xy) <- Sequences$ID
motif_xy <- motif_xy[sapply(motif_xy, length)>0] # remove sequences without motif
  motif_xy1 <- GRanges(names(unlist(motif_xy)), #Use sequence ID as seqnames
            IRanges(start(unlist(motif_xy)), end(unlist(motif_xy))),
            strand = mcols(unlist(motif_xy))$strand,
            score = mcols(unlist(motif_xy))$score)
  return(motif_xy1)
 })
names(motif_ix_pos2) <- c("Ohler1",
"DRE",
"TATAbox",
"INR",
"Ebox",
"Ohler6",
"Ohler7",
"Ohler8",
"DPE",
"MTE",
"TCT",
"DPE_extended",
"DPE_Kadonaga",
"BREu",
"BREd",
"DCE1",
"DCE2",
"DCE3",
"TC_17_Zabidi")
for (i in 1:length(motif_ix_pos2)) {
write.table( x = data.frame(motif_ix_pos2[[i]]), file = paste0("./",names(motif_ix_pos2)[i], "_gfzf.csv"), sep=",", col.names=TRUE, row.names=FALSE, quote=FALSE )
}
print("done csv")
df = read.table("coordinates_chro.txt",sep=",",header=TRUE,colClasses = c("character", "numeric", "numeric", "character"))
df$start <- df$start+1
seqinfo1= Dmelanogaster@seqinfo
Sequences <- with(df, GRanges(seqnames = Rle(chr), ranges = IRanges(start, end), strand = strand,seq=seq),seqinfo=seqinfo1)
Sequences$ID <- paste(Sequences@seqnames, Sequences@ranges@start, Sequences@ranges@start+Sequences@ranges@width-1,sep="_")
motif_ix <- matchMotifs(t1$Pwms_perc,
                        Sequences$seq,
                         genome = "BSgenome.Dmelanogaster.UCSC.dm3", p.cutoff = 1e-4, bg="genome", out = "positions")
print(motif_ix)
motif_ix_pos2 <- lapply(motif_ix, function(motif_xy){
names(motif_xy) <- Sequences$ID
motif_xy <- motif_xy[sapply(motif_xy, length)>0] # remove sequences without motif
  motif_xy1 <- GRanges(names(unlist(motif_xy)), #Use sequence ID as seqnames
            IRanges(start(unlist(motif_xy)), end(unlist(motif_xy))),
            strand = mcols(unlist(motif_xy))$strand,
            score = mcols(unlist(motif_xy))$score)
  return(motif_xy1)
 })
names(motif_ix_pos2) <- c("Ohler1",
"DRE",
"TATAbox",
"INR",
"Ebox",
"Ohler6",
"Ohler7",
"Ohler8",
"DPE",
"MTE",
"TCT",
"DPE_extended",
"DPE_Kadonaga",
"BREu",
"BREd",
"DCE1",
"DCE2",
"DCE3",
"TC_17_Zabidi")
for (i in 1:length(motif_ix_pos2)) {
write.table( x = data.frame(motif_ix_pos2[[i]]), file = paste0("./",names(motif_ix_pos2)[i], "_chro.csv"), sep=",", col.names=TRUE, row.names=FALSE, quote=FALSE )
}
print("done csv")
df = read.table("coordinates_mof.txt",sep=",",header=TRUE,colClasses = c("character", "numeric", "numeric", "character"))
df$start <- df$start+1
seqinfo1= Dmelanogaster@seqinfo
Sequences <- with(df, GRanges(seqnames = Rle(chr), ranges = IRanges(start, end), strand = strand,seq=seq),seqinfo=seqinfo1)
Sequences$ID <- paste(Sequences@seqnames, Sequences@ranges@start, Sequences@ranges@start+Sequences@ranges@width-1,sep="_")
motif_ix <- matchMotifs(t1$Pwms_perc,
                        Sequences$seq,
                         genome = "BSgenome.Dmelanogaster.UCSC.dm3", p.cutoff = 1e-4, bg="genome", out = "positions")
print(motif_ix)
motif_ix_pos2 <- lapply(motif_ix, function(motif_xy){
names(motif_xy) <- Sequences$ID
motif_xy <- motif_xy[sapply(motif_xy, length)>0] # remove sequences without motif
  motif_xy1 <- GRanges(names(unlist(motif_xy)), #Use sequence ID as seqnames
            IRanges(start(unlist(motif_xy)), end(unlist(motif_xy))),
            strand = mcols(unlist(motif_xy))$strand,
            score = mcols(unlist(motif_xy))$score)
  return(motif_xy1)
 })
names(motif_ix_pos2) <- c("Ohler1",
"DRE",
"TATAbox",
"INR",
"Ebox",
"Ohler6",
"Ohler7",
"Ohler8",
"DPE",
"MTE",
"TCT",
"DPE_extended",
"DPE_Kadonaga",
"BREu",
"BREd",
"DCE1",
"DCE2",
"DCE3",
"TC_17_Zabidi")
for (i in 1:length(motif_ix_pos2)) {
write.table( x = data.frame(motif_ix_pos2[[i]]), file = paste0("./",names(motif_ix_pos2)[i], "_mof.csv"), sep=",", col.names=TRUE, row.names=FALSE, quote=FALSE )
}
print("done csv")

