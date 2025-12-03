# How to Revert Time Management Fix

If you need to revert the time management changes made to improve search depth:

## Changes Made

1. **Moved setup work before time budget starts** - Voronoi, move ordering, and scenario building now happen BEFORE starting the 8-second time budget
2. **Added time caching** - Reduces callback overhead by caching `time_left()` calls
3. **Reduced safety margins** - Changed from 0.1s to 0.05s to use more of the time budget
4. **Less frequent time checks** - Check time every 3 moves instead of every move in inner loops

## To Revert

Restore from backup:
```bash
cd 3600-agents/CPPHikaru_3
cp search.cpp.backup search.cpp
```

Then rebuild:
```bash
make clean && make
# Or for Linux .so:
./build_linux_so.sh
```

## What Was Wrong Before

- Setup overhead (Voronoi, move ordering, trap scenarios) was eating into the 8-second budget
- Time checks were too frequent (calling Python callback through ctypes many times)
- Safety margins were too conservative (stopping with 0.1s left)
- Result: Only reaching depth 6-7 instead of target depths 11-16-22

## Expected Improvement

With these fixes, you should see:
- Average depth 10-12 for moves 1-10 (target: 11)
- Average depth 14-16 for moves 11-20 (target: 16)
- Average depth 20-22 for moves 21-30 (target: 22)
- Average depth 10-11 for moves 31-40 (target: 11)

