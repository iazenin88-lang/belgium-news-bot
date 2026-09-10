# Supabase Cron scheduler

The full news pipeline is started by the Supabase Cron job
"full-news-pipeline-dispatch". Supabase invokes the GitHub Actions
"workflow_dispatch" API; GitHub remains the executor for collector, analyzer,
and notifier.

Text corrections use the same Vault token to dispatch
`editorial_correction.yml` immediately after an editor comment. That workflow
runs only the correction for the supplied feedback id and sends the new
revision back to the editor chat; publication still requires the editor's
button.

## Schedule

The job runs at minute 07 and 37 during UTC hours 07–23:

    7,37 7-23 * * *

The database function applies the exact 09:07–00:37 Europe/Brussels window,
including the CET/CEST transition. Calls outside that window are logged and do
not start GitHub Actions.

## Secret setup

Create a fine-grained GitHub token limited to this repository with Actions
workflow dispatch permission. Store it in Supabase Vault; never commit or send
the value in chat:

    select vault.create_secret(
      '<GITHUB_TOKEN>',
      'github_actions_token',
      'Fine-grained token for the full news pipeline dispatcher'
    );

The function reads only vault.decrypted_secrets at execution time. After
rotation, replace the Vault secret with the same name.

The immediate correction dispatcher writes decisions to
`private.editorial_correction_dispatch_log`. A missing token leaves the
correction pending, so the regular analyzer can recover it on its next run.

## Monitoring

Supabase Cron history:

    select *
    from cron.job_run_details
    where jobid = (
      select jobid
      from cron.job
      where jobname = 'full-news-pipeline-dispatch'
    )
    order by start_time desc
    limit 20;

Dispatcher decisions:

    select *
    from private.pipeline_dispatch_log
    order by invoked_at desc, id desc
    limit 50;

The queued row contains the pg_net request id. GitHub workflow runs can be
checked in the Actions tab or with the repository workflow-runs API.

## Rollback

Deactivate or remove the job before restoring the previous GitHub schedule:

    select cron.unschedule('full-news-pipeline-dispatch');
