# Get Delay 

Format in `.log` files:

```python
'[FRAME_INFO] T: %d, dec_dur: %d, Bytes: %d', frame_timestamp(frame_id), delay, size_of_frame
```

## TODO

- [x] encode [enc_dur]
- [ ] (pacer)
- [ ] send
- [x] jitter [jit_dur]
- [x] decode [dec_dur]
- [ ] render