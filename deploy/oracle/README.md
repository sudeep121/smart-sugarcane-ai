# Hosting Smart Sugarcane AI on Oracle Cloud Always Free

The result is a permanent HTTPS address that works on **any device on any
network** — mobile data included — running the real trained models rather than
demo heuristics.

## Why this host

Vercel caps Python functions at 250 MB; TensorFlow alone exceeds that, which is
why the Vercel deployment has always served demo results. Render's free tier
gives 512 MB of RAM and 0.1 CPU, where TensorFlow is killed on import. Hugging
Face Docker Spaces stopped being free — they now require a paid plan.

Oracle's Always Free tier gives **2 OCPU and 12 GB of RAM, permanently**, with
no card charge and no expiry. That is roughly six times the memory the
application needs.

Two things to know before starting:

- **Oracle halved this tier on 15 June 2026**, from 4 OCPU / 24 GB, without
  announcing it. 2 OCPU / 12 GB is still ample here, but older guides you find
  will quote the larger figures.
- **A credit card is required for identity verification.** Always Free
  resources are not charged, but Oracle will not create the account without it.

## What you need

| | |
|---|---|
| Oracle Cloud account | free, card required for verification |
| A free DuckDNS subdomain | for HTTPS — [duckdns.org](https://www.duckdns.org) |
| About 45 minutes | most of it waiting for the build |

HTTPS is not optional. Without it Android refuses to install the app to the
home screen, service workers will not register, and browsers warn on every
visit. A DuckDNS subdomain costs nothing and lets Caddy obtain a real Let's
Encrypt certificate automatically.

---

## 1. Create the instance

1. Sign up at [cloud.oracle.com](https://cloud.oracle.com). Choose a home
   region close to you — **it cannot be changed later**.
2. **Compute → Instances → Create instance**
3. **Image and shape → Change shape**
   - Shape series: **Ampere** (this is the ARM one with the free 12 GB)
   - Shape: `VM.Standard.A1.Flex`
   - OCPUs: **2**, Memory: **12 GB**
   - Confirm it is labelled **Always Free eligible**
4. Image: **Ubuntu 22.04** or **24.04**
5. Under Networking, tick **Assign a public IPv4 address**
6. Save the private key when prompted. Without it you cannot log in.
7. Create.

> **"Out of host capacity"** is common — Ampere is heavily oversubscribed.
> It is not a mistake on your part. Retry periodically, or try a different
> availability domain or region. Some people script the retry.

## 2. Open the ports in the cloud firewall

Oracle has two separate firewalls and both must allow traffic. This is the
cloud-side one; `setup.sh` handles the instance-side one.

**Networking → Virtual Cloud Networks →** your VCN **→ Security Lists →**
default list **→ Add Ingress Rules**, twice:

| Source CIDR | IP Protocol | Destination Port |
|---|---|---|
| `0.0.0.0/0` | TCP | `80` |
| `0.0.0.0/0` | TCP | `443` |

## 3. Point a hostname at it

1. Go to [duckdns.org](https://www.duckdns.org), sign in with Google/GitHub.
2. Create a subdomain, e.g. `smart-sugarcane`.
3. Set its IP to your instance's **public IPv4 address** and update.

Your address is now `smart-sugarcane.duckdns.org`.

## 4. Deploy

SSH in (the username is `ubuntu` for Ubuntu images):

```bash
ssh -i /path/to/your-key.key ubuntu@YOUR_PUBLIC_IP
```

Then:

```bash
git clone https://github.com/YOUR_USERNAME/smart-sugarcane-ai.git
cd smart-sugarcane-ai
bash deploy/oracle/setup.sh
```

The first run installs Docker, opens the instance firewall, adds swap, writes
a `.env` with a generated `SECRET_KEY`, and stops.

Edit `deploy/oracle/.env` and set `DOMAIN` to your DuckDNS hostname, then run
the script again:

```bash
nano deploy/oracle/.env      # set DOMAIN=smart-sugarcane.duckdns.org
bash deploy/oracle/setup.sh
```

The build takes **15–25 minutes** on two ARM cores — almost all of it
installing TensorFlow. When it finishes:

**`https://smart-sugarcane.duckdns.org`**

## 5. Confirm the models actually loaded

The whole point is not to be in demo mode:

```bash
curl -s https://YOUR_DOMAIN/api/soil/model-status | jq '.trained_model_available, .model_source'
```

Expect `true` and `"trained_model"`. If you see `false` and
`"demo_heuristic"`, TensorFlow did not install — check `docker compose logs app`.

---

## Day-to-day

```bash
cd ~/smart-sugarcane-ai/deploy/oracle
docker compose logs -f app        # follow logs
docker compose restart            # restart
docker compose down               # stop
git -C ~/smart-sugarcane-ai pull && docker compose up -d --build   # update
```

## Data

By default the app uses SQLite on the `appdata` volume, so accounts and saved
analyses survive container restarts and rebuilds — but not the VM being
terminated. For durability beyond the VM, set `DATABASE_URL` in `.env` to a
managed Postgres instance; the `psycopg` driver is already installed.

## Things that will bite you

**Port 80/443 unreachable although the Security List allows them.** Oracle's
Ubuntu images carry an iptables ruleset with a catch-all REJECT. `setup.sh`
inserts the allow rules above it, but if you rebuilt the rules by hand, check
`sudo iptables -L INPUT -n --line-numbers` and make sure your ACCEPT lines come
before the REJECT.

**Caddy cannot get a certificate.** The HTTP-01 challenge needs port 80 open
and DNS already pointing at the VM. Verify with
`getent hosts your.duckdns.org` from the VM, and check `docker compose logs caddy`.

**Idle reclamation.** Oracle may reclaim Always Free compute that sits unused.
Real traffic is the simplest protection; an uptime pinger every few minutes
also works and keeps the models warm.

**The first request after a restart is slow.** Loading TensorFlow and three
models takes over a minute. That is startup cost, not a hang — the health
check allows five minutes for it.

## Pointing the Android app here

The APK no longer has a baked-in address. On the login screen tap
`Server: … — change`, enter your DuckDNS hostname, and save. Because the host
is HTTPS, "Add to Home Screen" also produces a proper installed app rather
than a bookmark.
