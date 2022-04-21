https://unix.stackexchange.com/questions/67055/how-to-scp-with-regular-expressions

Just ran

```
scp -r "ahernandez@txe1-login.mit.edu:/home/gridsan/ahernandez/git/sims_resnets_small_all_except1111/myScript.*" .
```

and
 
```
scp -r ahernandez@txe1-login.mit.edu:/home2/gridsan/ahernandez/git/Plato/rumen/myScript.sh.log-14471538-1 .
```

Make sure that the thigns inside the logs to analyze are ONLY logs
