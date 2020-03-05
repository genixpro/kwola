import React from 'react';
import Loader from '../utility/Loader';
import HelperText from '../utility/helper-text';
import Pagination from '../uielements/pagination';
import { DemoWrapper } from '../utility/papersheet';
import { per_page } from '../../redux/githubSearch/sagas';
import {
  GithubResultListStyleWrapper,
  GithubResultStyleWrapper,
  PapersheetStyle,
  StarIcon,
} from './githubResult.style';

function SearchList(result) {
  return (
    <GithubResultListStyleWrapper className="GithubResultList">
      {result.map(item => {
        const onClick = () => {
          window.open(item.html_url, '_blank');
        };
        const updateDate = new Date(item.updated_at).toDateString();
        return (
          <PapersheetStyle key={item.id} className="SingleRepository">
            <div className="titleAndLanguage">
              <h3>
                <a href="#!" onClick={onClick}>{`${item.full_name} `}</a>
              </h3>

              {item.language ? (
                <span className="language">{item.language}</span>
              ) : (
                ''
              )}
              {item.stargazers_count ? (
                <span className="totalStars">
                  <StarIcon>grade</StarIcon>
                  {item.stargazers_count}
                </span>
              ) : (
                ''
              )}
            </div>
            {item.description ? <p>{item.description}</p> : ''}
            <span className="updateDate">Updated on {updateDate}</span>
          </PapersheetStyle>
        );
      })}
    </GithubResultListStyleWrapper>
  );
}
const GitResult = ({ GitSearch, onPageChange }) => {
  const { searcText, result, loading, error, page, total_count } = GitSearch;
  if (!searcText) {
    return <div />;
  }
  if (loading) {
    return (
      <DemoWrapper>
        <Loader />
      </DemoWrapper>
    );
  }
  if (error || !total_count) {
    return <HelperText text="THERE ARE SOME ERRORS" />;
  }
  if (result.length === 0) {
    return <HelperText text="No Result Found" />;
  }
  const visibleItem = total_count > 1000 ? 1000 : total_count;
  const pageCount = Math.floor(visibleItem / per_page);
  return (
    <GithubResultStyleWrapper className="GithubSearchResult">
      <p className="TotalRepository">
        <span>{`${total_count}`} repository results</span>
      </p>
      <DemoWrapper>{SearchList(result)}</DemoWrapper>
      <div className="githubSearchPagination">
        <Pagination
          defaultCurrent={page}
          total={pageCount}
          onChange={page => {
            onPageChange(searcText, page);
          }}
        />
      </div>
    </GithubResultStyleWrapper>
  );
};

export default GitResult;
