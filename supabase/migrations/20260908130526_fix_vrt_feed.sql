do $$
begin
  update public.sources
  set url = 'https://www.vrt.be/vrtnws/nl.rss.articles.xml'
  where name = 'VRT NWS';

  if not found then
    raise exception 'VRT NWS source was not found';
  end if;
end;
$$;
