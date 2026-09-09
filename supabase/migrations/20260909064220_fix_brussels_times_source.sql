do $$
declare
  updated_rows integer;
begin
  update public.sources
  set url = 'https://www.brusselstimes.com/google-news-sitemap.xml'
  where name = 'Brussels Times';

  get diagnostics updated_rows = row_count;
  if updated_rows <> 1 then
    raise exception 'Expected one Brussels Times source, updated %', updated_rows;
  end if;
end
$$;
