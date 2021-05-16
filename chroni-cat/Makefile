CC = g++
PKG_CONFIG_LIBRARIES = opencv4
CFLAGS   = -Wall -Wextra -std=gnu++17 -O3
INCLUDES = $(shell pkg-config --cflags $(PKG_CONFIG_LIBRARIES))
LDLIBS   = $(shell pkg-config --libs   $(PKG_CONFIG_LIBRARIES))
LDFLAGS  = 

CPPS = $(wildcard *.cpp)
HPPS = $(wildcard *.hpp)
OBJS = $(CPPS:.cpp=.o)
OUT  = chronicat

$(OUT): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(CFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CC) $(LDFLAGS) -c -o $@ $< $(CFLAGS) $(INCLUDES)

.PHONY: clean distclean

clean:
	rm -f $(OBJS)

distclean:
	rm -f $(OBJS) $(OUT)
