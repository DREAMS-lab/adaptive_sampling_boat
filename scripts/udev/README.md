# udev rules for stable serial port names

The boat has three USB-serial devices that get renumbered (`/dev/ttyUSB0`, `ttyUSB1`, `ttyUSB2`) depending on the order they power up. This causes MAVROS to occasionally try to talk to the sonde and the sonar driver to open the Pixhawk, both of which silently fail.

## One-time setup per Odroid

1. Plug in **only the Pixhawk** and note which `/dev/ttyUSBn` it claims. Grab its vendor/product/serial:

   ```bash
   udevadm info --attribute-walk /dev/ttyUSBn | grep -E 'idVendor|idProduct|serial' | head -3
   ```

2. Unplug the Pixhawk, plug in **only the sonde**, repeat. Then the sonar.

3. Edit `99-boat-serial.rules` — replace each `<FILL_IN_..._SERIAL>` placeholder with the serial from step 1/2.

4. Install:

   ```bash
   sudo cp 99-boat-serial.rules /etc/udev/rules.d/
   sudo udevadm control --reload
   sudo udevadm trigger
   ```

5. Verify — plug all three in, in any order, and confirm:

   ```bash
   ls -l /dev/boat_*
   # boat_fc -> ttyUSB?
   # boat_sonar -> ttyUSB?
   # boat_sonde -> ttyUSB?
   ```

All config files under `Boat_control/config/` reference `/dev/boat_*` so once the symlinks exist the whole stack is order-independent.
