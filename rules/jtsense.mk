# Copyright 2013. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.




jtsensesrcs := $(wildcard $(srcdir)/jtsense/*.c)
jtsenseobjs := $(jtsensesrcs:.c=.o)

.INTERMEDIATE: $(jtsenseobjs)

lib/libjtsense.a: libjtsense.a($(jtsenseobjs))



