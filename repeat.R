x <- read.table("L5repeats.txt")
y <- read.table("L50repeats.txt")
z <- read.table("L200repeats.txt")
q <- read.table("L1000repeats.txt")

png("repeats.png")

plot(q$V1) # plot this b/c it has biggest y range
points(q$V2, pch = "+")

points(z$V1, col = "purple") # plot this b/c it has biggest y range
points(z$V2, pch = "+", col = "purple")

points(x$V1, col = "red") # plot this b/c it has biggest y range
points(x$V2, pch = "+", col = "red")

points(y$V1, col = "blue") # plot this b/c it has biggest y range
points(y$V2, pch = "+", col = "blue")

dev.off()
